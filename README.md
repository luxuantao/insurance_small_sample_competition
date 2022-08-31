# 代码说明

## 整体思路

1. 沿用baseline中的方法，先在opensource数据上进行微调，然后再在instruction数据上进行微调

2. 在instruction测试数据上进行数据增强（训练数据也可以用同样的方式增强，但是实测下来没有用），具体为：在模型inference的时候对verbalizer的出现顺序进行了扰动，使得同一条数据被inference多次，得到不同的预测结果，然后进行投票得出最终的预测结果

3. 模型为单模单折，经过int8量化的mt5-large模型，总大小为1.4GB，符合要求

## 复现步骤

1. 安装依赖 `pip install -r requirements.txt` ，安装完成后主要检查一下onnx库的版本是否为1.11.0，onnxruntime库的版本是否为1.11.1，如果后续操作中出现问题，请对齐版本，没问题就忽略。此外，我用的python版本为3.7

2. （**重要**）对onnx库中的源代码进行修改，这是因为onnx库的代码目前存在一定bug，大于大模型的量化会报错，具体路径需要换成你们机器上python解释器的路径，下面是一个示例：
   
   `xxx/python3.7/site-packages/onnxruntime/quantization/onnx_quantizer.py`
   
   在该python文件第32行左右的位置有如下的代码：
   
   ```python
   if not ('DisableShapeInference' in self.extra_options and self.extra_options[' DisableShapeInference']):
       model = onnx.shape_inference.infer_shapes(model)
   ```
   
   修改为:
   
   ```python
   if not ('DisableShapeInference' in self.extra_options and self.extra_options[' DisableShapeInference']):
       pass
   ```

3. `bash pretrain_mt5_large.sh` 在opensource数据集上预训练模型，训练中显存占用约为65GB，我训练用的GPU是单卡A100

4. `python get_instruction_data.py` 将instruction数据整理为统一格式

5. `python get_ensemble_test_data_b.py` 对测试数据进行数据增强

6. `bash direct_train_mt5_large.sh` 在instruction数据上进行训练，注意该脚本文件中：`--model_path="./results/xxx/checkpoint-50052"` 该命令行参数需要换成第三步训练得到的模型的保存路径，如果完全复现的话，第三步执行结束后，该路径中会出现 `checkpoint-50052` 的权重文件

7. `python onnx_main.py` 模型导出为onnx格式，并进行int8量化，并最终进行预测，注意第297行 `generate_onnx_representation(pretrained_version='./results/xxx/checkpoint-875', output_prefix='./results/onnx_origin_model/mt5-large')` ，其中的`pretrained_version` 参数需要换成第六步训练得到的模型的保存路径，如果完全复现的话，第六步执行结束后，该路径中会出现 `checkpoint-875` 的权重文件。预测过程可能会比较慢，约40分钟，因为我编写的代码中，是用CPU预测的而没有用到GPU，而且是一条条sample进行预测，没有使用batch的方式（这样是符合真实线上部署环境的），并且由于我对测试数据进行了样本增强，所以样本数目约为没增强前的四倍。其实我的模型推理速度是不慢的，因为经过了int8量化，而且用了onnx进行推理加速。如果别人的模型也只用CPU进行推理，肯定没有我的快。

8. `python get_final_result.py` 对最终结果进行后处理，最终得到的预测结果文件为`answer.json` 

为了方便复现，我也将最终用到的预测模型上传到了邮件的附件中，下载后将其放在`results` 路径下，该模型可以复现出线上B榜的最佳成绩
