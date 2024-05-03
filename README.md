---
title: Ascend平台上的Llama2部署记录
refs: 
  - https://github.com/zzudongxiang/ascend.llama2
  - https://gitee.com/zzudongxiang/ascend.llama2
---

# 1. 环境配置

初上手华为昇腾910B芯片的NPU，根据学习需要在上面尝试跑通Llama2的推理、评测、训练等任务，但是Llama2的原生代码无法直接放在Ascend平台上运行，需要对代码进行一定的魔改适配Ascend平台，根据调研，计划使用Mindspore框架对Llama2的原生模型进行修改。

由于所使用的服务器要求使用docker跑自己的应用，因此需要从创建docker容器，挂载相关的硬件设备和驱动开始，需要注意的是使用docker挂载的只有Ascend的Driver和Firmware，相关的Toolkit和Kernel等组件在docker内根据需要自己安装对应的版本即可。

创建容器和配置基础环境的过程请参考：[Gitee](https://gitee.com/zzudongxiang/ascend.docker)、[Github](https://github.com/zzudongxiang/ascend.docker)

# 2. Llama2推理

Mindformers自带Llama2的模型，可以直接使用API调用Llama2，在缺失权重文件和Token文件时会自动下载对应的缺失文件

## 2.1 API接口调用

配置单机多卡或多机多卡需要额外的配置服务器的信息，这里先使用单机单卡推理任务验证环境是否配置成功，在任意文件夹创建如下Python文件，并命名为`llama2.py`

可以根据如下几种API调用方法中的任意一种进行单机单卡推理任务，然后使用`python llama2.py`命令运行，使用API的接口调用Llama2需要提前使用pip安装Mindformers

### 2.1.1 基于AutoClass

```python
# llama2.py
import mindspore
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# model的实例化有以下两种方式，选择其中一种进行实例化即可
# 1. 直接根据默认配置实例化
model = AutoModel.from_pretrained('llama2_7b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('llama2_7b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
# config.xxx = xxx                      # 根据需求自定义修改其余模型配置
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

inputs = tokenizer("I love Beijing, because")["input_ids"]
# 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
outputs = model.generate(inputs, max_new_tokens=30, do_sample=False)
response = tokenizer.decode(outputs)
print(response)
# ['<s>I love Beijing, because it’s a city that is constantly changing. I have been living here for 10 years and I have seen the city change so much.I']
```

### 2.1.2 基于Pipeline

```python
# llama2.py
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline("text_generation", model='llama2_7b', max_length=30)
pipeline_result = pipeline_task("I love Beijing, because", do_sample=False)
print(pipeline_result)
# [{'text_generation_text': ['<s>I love Beijing, because it’s a a city that is constantly changing. I have been living here for 10 years and I have']}]
```

## 2.2 Mindspore推理

使用API的方式调用Llama2推理性能较差，单机单卡在7B模型上大概只有1.3tokens/s的生成速度，通过下载Mindformers源码，并从源码编译运行Llama2模型将获得更快的推理速度，推理速度大概在10tokens/s左右。

无论是基于Generate还是基于Pipeline的推理，文件的前半部分都是按照如下内容编写：

```python
import mindspore as ms
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool, get_real_rank
from mindformers.trainer.utils import get_last_checkpoint

# 初始化输入内容
inputs = [
    "I love Beijing, because",
    "LLaMA is a",
    "Huawei is a company that",
    ]
yaml_file="/path/to/yaml_file.yaml"
use_past = False
seq_length = 512
checkpoint_path = "/path/ro/checkpoint.ckpt"
model_type = "llama2_7b"

# 读取配置文件
config = MindFormerConfig(yaml_file)
print(config)

# 初始化环境
tokenizer = AutoTokenizer.from_pretrained(model_type)
init_context(use_parallel=config.use_parallel, context_config=config.context, parallel_config=config.parallel)
model_config = LlamaConfig(**config.model.model_config)
model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
model_config.use_past = use_past
model_config.seq_length = seq_length

# 加载CheckPoint文件
if checkpoint_path and not config.use_parallel:
    model_config.checkpoint_name_or_path = checkpoint_path
print(f"config is: {model_config}")

# 构建模型
model = LlamaForCausalLM(model_config)
model.set_train(False)

# 并行运行
if config.use_parallel:
    ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
    ckpt_path = get_last_checkpoint(ckpt_path)
    print("ckpt path: %s", str(ckpt_path))
    warm_up_model = Model(model)
    warm_up_model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
    checkpoint_dict = load_checkpoint(ckpt_path)
    not_load_network_params = load_param_into_net(model, checkpoint_dict)
    print("Network parameters are not loaded: %s", str(not_load_network_params))
```

### 2.2.1 基于Generate

新建`llama2.py`文件，将以上内容和以下内容合并为一个文件，并根据实际情况修改python文件中的变量参数，然后直接使用python脚本运行即可。

```python
# 导入上一小节的Python代码
...
# 获取输出
inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"]
outputs = model.generate(inputs_ids,
                         max_length=model_config.max_decode_length,
                         do_sample=model_config.do_sample,
                         top_k=model_config.top_k,
                         top_p=model_config.top_p)
for output in outputs:
    print(tokenizer.decode(output))
```

### 2.2.2 基于Pipeline

基于Pipeline的推理调用方法，用法与上一节相似，也需要将公共部分的代码复制到该文件的前半部分。

```python
# 导入上一小节的Python代码
...
# 获取输出
text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
outputs = text_generation_pipeline(inputs,
                                   max_length=model_config.max_decode_length,
                                   do_sample=model_config.do_sample,
                                   top_k=model_config.top_k,
                                   top_p=model_config.top_p)
for output in outputs:
    print(output)
```

