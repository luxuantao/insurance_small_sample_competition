from transformers import (
    T5Tokenizer,
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoConfig,
    BertTokenizer, 
    BartForConditionalGeneration,
    BartConfig,
    MBartTokenizer,
    MBartForConditionalGeneration
)
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from .modeling_cpt import CPTForConditionalGeneration, CPTConfig


def init_baseline_model(model_args, model_name, data_args, special_tokens=[]):
    print("init {} model from {}...".format(model_name, model_args.model_name_or_path))
    if model_name == "T5":
        tokenizer = T5Tokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=True,
            max_length=1024,
            truncation=True,
            additional_special_tokens=special_tokens,
        )
        config = T5Config.from_pretrained(model_args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
        model.config.max_length = data_args.val_max_target_length
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == "MT5":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=True,
            max_length=1024,
            truncation=True,
            additional_special_tokens=special_tokens,
        )
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = MT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
        model.config.max_length = data_args.val_max_target_length
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == "BART":
        tokenizer = BertTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=True,
            max_length=1024,
            truncation=True,
            additional_special_tokens=special_tokens,
        )
        model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        model.config.max_length = data_args.val_max_target_length
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == "MBART":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=True,
            max_length=1024,
            truncation=True,
            additional_special_tokens=special_tokens,
            src_lang='zh_CN',
            tgt_lang='zh_CN'
        )
        model = MBartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        model.config.max_length = data_args.val_max_target_length
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == "BART-LARGE":
        tokenizer = BertTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=True,
            max_length=1024,
            truncation=True,
            additional_special_tokens=special_tokens,
        )
        config = BartConfig.from_pretrained(model_args.model_name_or_path)
        config.decoder_layers = 11
        model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
        model.config.max_length = data_args.val_max_target_length
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == "CPT":
        tokenizer = BertTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=True,
            max_length=1024,
            truncation=True,
            additional_special_tokens=special_tokens,
        )
        model = CPTForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        model.config.max_length = data_args.val_max_target_length
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == "CPT-LARGE":
        tokenizer = BertTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=True,
            max_length=1024,
            truncation=True,
            additional_special_tokens=special_tokens,
        )
        config = CPTConfig.from_pretrained(model_args.model_name_or_path)
        config.decoder_layers = 3
        model = CPTForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
        model.config.max_length = data_args.val_max_target_length
        model.resize_token_embeddings(len(tokenizer))
    else:
        raise NotImplementedError("You can implement {} by yourself".format(model_name.upper()))

    return tokenizer, model
