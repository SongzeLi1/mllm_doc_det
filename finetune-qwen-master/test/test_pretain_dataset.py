# %%
import os

# -*- coding: utf-8 -*-
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.append("..")

from swift.llm import get_model_tokenizer
from swift.llm.template import TEMPLATE_MAPPING

# %%
# Obtain the model and template
model_dir = "Qwen/Qwen2.5-VL-3B-Instruct"
model, processor = get_model_tokenizer(model_dir, use_hf=True)

# %%
from finetune.datasets import ForensicTemplate

# 实例化模板
if model is None:
    template_name = "qwen2_5_vl"
else:
    template_name = model.model_meta.template
template = ForensicTemplate(processor, TEMPLATE_MAPPING[template_name])
template.set_mode("train")

from finetune.datasets import CustomDataset

# 其余代码不变
json_path = "/home/yuyangxin/data/swift-demo/resource/datasets/without_instruct/CASIAv2.json"
conversation_templates = "/pubdata/yuyangxin/swift-demo/finetune/templates/pretrain_template.txt"
dataset = CustomDataset(json_path, template, conversation_templates, num_new_tokens=32, add_augmentation=True)

print("Dataset size:", len(dataset))
print("Dataset[0]:", dataset[0])
print("Dataset[1]:", dataset[1])
