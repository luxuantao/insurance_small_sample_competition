import torch
import torch.nn.functional as F
from tqdm import trange
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from onnxruntime.quantization import quantize_dynamic
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import json
from tqdm import tqdm


class CombinedDecoder(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config

    def forward(self, input_ids, encoder_hidden_states):
        decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)[0] * \
            (self.config.d_model ** -0.5)
        return self.lm_head(decoder_output)


class SimplifiedT5Encoder(torch.nn.Module):
    """ Creation of a class to output only the last hidden state from the encoder """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, *input, **kwargs):
        return self.encoder(*input, **kwargs)[0]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Function created by Thomas Wolf of the huggingface team
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class GenerativeT5(torch.nn.Module):
    """ This wrapper utility function implements a single beam search to generate efficiently text.
        A lot of the credit goes to the huggingface team and its chief scientist Thomas Wolf whose implementation I based
        myself off.

        Args:
            encoder: huggingface encoder or onnx session for the encoder of T5. Can be obtained with the
                create_t5_encoder_decoder utility function for pytorch, see examples below.
            decoder_with_lm_head: decoder with language model head on top. Can be obtained with the
                create_t5_encoder_decoder utility function for pytorch, see examples below.
            tokenizer: huggingface tokenizer
            onnx (bool): whether to use onnx or the default pytorch
            cuda (bool): whether to use cuda or the cpu

        Examples:
            For pytorch:
            >>> from transformers import T5Tokenizer
            >>> from onnxt5 import create_t5_encoder_decoder, GenerativeT5
            >>> pretrained_model = 't5-base' # This can be a pretrained version, or the path to a huggingface model
            >>> simplified_encoder, decoder_with_lm_head = create_t5_encoder_decoder(pretrained_model)
            >>> tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
            >>> generative_t5 = GenerativeT5(simplified_encoder, decoder_with_lm_head, tokenizer)
            >>> generative_t5('translate English to French: I was a victim of a series of accidents.', 16, temperature=0.)[0]
            >>> # Output: "Je suis victime d'une série d'accidents."

            For onnx:
            >>> from transformers import T5Tokenizer
            >>> from onnxruntime import InferenceSession
            >>> from onnxt5 import GenerativeT5
            >>> # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
            >>> # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
            >>> # based on the build flags) when instantiating InferenceSession.
            >>> # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
            >>> # InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
            >>> decoder_sess = InferenceSession('~/t5-decoder-with-lm-head.onnx')
            >>> encoder_sess = InferenceSession('~/t5-encoder.onnx')
            >>> tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
            >>> generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
            >>> generative_t5('translate English to French: I was a victim of a series of accidents.', 16, temperature=0.)[0]
            >>> # Output: "Je suis victime d'une série d'accidents."

    """

    def __init__(self, encoder, decoder_with_lm_head, tokenizer, onnx=False, cuda=False):
        super().__init__()
        self.encoder = encoder
        self.decoder_with_lm_head = decoder_with_lm_head
        self.tokenizer = tokenizer
        self.onnx = onnx
        self.cuda = cuda

    def forward(self, prompt, max_length, temperature=1., repetition_penalty=1., top_k=50, top_p=0, max_context_length=512):
        """ Forward function to generate text after a prompt
            Args:
                prompt: str to run (don't forget to add at the beginning the task to run such as "summarize:"
                        or "translate English to German:"
                max_context_length: maximum number of tokens to use as context

        """
        with torch.no_grad():
            new_tokens = torch.tensor((), dtype=torch.long)
            new_logits = []
            generated = torch.tensor(self.tokenizer(prompt)['input_ids'])[:max_context_length - 1].unsqueeze(0)
            if self.cuda and not self.onnx:
                generated = generated.cuda()

            # Getting encoder past
            if self.onnx:
                encoder_outputs_prompt = self.encoder.run(None, {"input_ids": generated.cpu().numpy()})[0]
            else:
                encoder_outputs_prompt = self.encoder(generated)

            # The sequence now needs to start with a
            generated = torch.zeros((1, 1), dtype=torch.long)
            if self.cuda and not self.onnx:
                generated = generated.cuda()

            for _ in range(max_length):
                if self.onnx:
                    outputs = torch.tensor(self.decoder_with_lm_head.run(None, {"input_ids": generated.cpu().numpy(),
                                                                                "encoder_hidden_states": encoder_outputs_prompt})[0][0])
                else:
                    outputs = self.decoder_with_lm_head(input_ids=generated,
                                                        encoder_hidden_states=encoder_outputs_prompt)[0]
                next_token_logits = outputs[-1, :] / (temperature if temperature > 0 else 1.0)
                if int(next_token_logits.argmax()) == 1:
                    break
                new_logits.append(next_token_logits)
                for _ in set(generated.view(-1).tolist()):
                    next_token_logits[_] /= repetition_penalty
                if temperature == 0:  # greedy sampling:
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                else:
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                new_tokens = torch.cat((new_tokens, next_token), 0)

            return self.tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False), new_logits


def create_t5_encoder_decoder(pretrained_version='google/mt5-large'):
    """ Generates an encoder and a decoder model with a language model head from a pretrained huggingface model

    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5

    Returns:
        simplified_encoder: pytorch t5 encoder with a wrapper to output only the hidden states
        decoder_with_lm_head: pytorch t5 decoder with a language modeling head
    """

    # T5 is an encoder / decoder model with a language modeling head on top.
    # We need to separate those out for efficient language generation
    model = MT5ForConditionalGeneration.from_pretrained(pretrained_version)

    return turn_model_into_encoder_decoder(model)


def turn_model_into_encoder_decoder(model):
    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head

    decoder_with_lm_head = CombinedDecoder(decoder, lm_head, model.config)
    simplified_encoder = SimplifiedT5Encoder(encoder)

    return simplified_encoder, decoder_with_lm_head


def generate_onnx_representation(pretrained_version=None, output_prefix=None, model=None):
    """ Exports a given huggingface pretrained model, or a given model and tokenizer, to onnx

    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5
        output_prefix (str): Path to the onnx file
    """
    if (pretrained_version is None or output_prefix is None) and model is None:
        print("You need to specify both pretrained_version (the pretrained model you wish to export) and output_prefix"
              "(the path you want to export to). Alternatively you can export a model you have in memory.")
        return
    if model is not None:
        # Transform model into encoder and decoder with lm head
        simplified_encoder, decoder_with_lm_head = turn_model_into_encoder_decoder(model)
    else:
        # Loading model_data
        simplified_encoder, decoder_with_lm_head = create_t5_encoder_decoder(pretrained_version)

    # Example sequence
    input_ids = torch.tensor([[42] * 10])

    # Exports to ONNX
    _ = torch.onnx.export(
        decoder_with_lm_head,
        (input_ids, simplified_encoder(input_ids)),
        f"{output_prefix}-decoder-with-lm-head.onnx",
        export_params=True,
        opset_version=12,
        input_names=['input_ids', 'encoder_hidden_states'],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'encoder_hidden_states': {0: 'batch', 1: 'sequence'},
            'hidden_states': {0: 'batch', 1: 'sequence'},
        },
        use_external_data_format=True
    )

    _ = torch.onnx._export(
        simplified_encoder,
        input_ids,
        f"{output_prefix}-encoder.onnx",
        export_params=True,
        opset_version=12,
        input_names=['input_ids'],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'encoder_hidden_states': {0: 'batch', 1: 'sequence'},
            'hidden_states': {0: 'batch', 1: 'sequence'},
        },
        use_external_data_format=True
    )


def quant():
    model_fp32 = './results/onnx_origin_model/mt5-large-encoder.onnx'
    model_quant = './results/onnx_quantize_model/quantize_mt5-large-encoder.onnx'
    quantized_model = quantize_dynamic(model_fp32, model_quant, use_external_data_format=True)

    model_fp32 = './results/onnx_origin_model/mt5-large-decoder-with-lm-head.onnx'
    model_quant = './results/onnx_quantize_model/quantize_mt5-large-decoder-with-lm-head.onnx'
    quantized_model = quantize_dynamic(model_fp32, model_quant, use_external_data_format=True)


def inference():
    encoder_sess = InferenceSession('./results/onnx_quantize_model/quantize_mt5-large-encoder.onnx')#, providers=['CUDAExecutionProvider'])
    decoder_sess = InferenceSession('./results/onnx_quantize_model/quantize_mt5-large-decoder-with-lm-head.onnx')#, providers=['CUDAExecutionProvider'])

    tokenizer = AutoTokenizer.from_pretrained('google/mt5-large')
    generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
    print(generative_t5('【就是我我有一个车啊就是这种商业险就是我现在已经卖掉了已经过户给人家他这个保险要不要更改呀因为这个保险还没到期吗】这句话的意图是什么？变更车辆信息/客户信息变更/投保人变更/报案信息修改/续期缴费方式变更', 200, temperature=0)[0])

    with open('./data/ensemble_instruction_test_b.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    preds = {}
    for each in tqdm(test_data):
        ans = generative_t5(each['instruction'], 200, temperature=0)[0]
        if ans == '无答案' or ans == '暂无答案':
            ans = ''
        preds[each['ID']] = ans
    # NER format
    for k, v in preds.items():
        if 'NER' in k:
            preds[k] = v.split(',')
    with open('./results/onnx_predict/ensemble_answer.json', 'w', encoding='utf-8') as f:
        json.dump(preds, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    print('导出为ONNX...')
    generate_onnx_representation(pretrained_version='./results/xxx/checkpoint-875', output_prefix='./results/onnx_origin_model/mt5-large')
    
    print('开始量化...')
    quant()
    
    print('开始推理...')
    inference()

#     /home/tiger/.local/lib/python3.7/site-packages/onnxruntime/quantization/onnx_quantizer.py