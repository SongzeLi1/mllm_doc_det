import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import fetch_image, process_vision_info
from swift.llm import (
    EncodePreprocessor,
    LazyLLMDataset,
    get_model_arch,
    get_model_tokenizer,
    get_multimodal_target_regex,
    get_template,
    load_dataset,
)
from transformers import AutoProcessor

from finetune.models import QwenForensicModel


def inference(image_path, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False):
    image = Image.open(image_path)
    image_local_path = "file://" + image_path
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"image": image_local_path},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]


# 1. 模型路径标准化加载
model_dir = "Qwen/Qwen2.5-VL-3B-Instruct"
# 2. 显存优化配置
model = QwenForensicModel.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    use_cache=True,
)
# 3. 专用处理器初始化（支持动态分辨率）
processor = AutoProcessor.from_pretrained(model_dir, use_fast=True)

image_path = "/pubdata/yuyangxin/swift-demo/resource/test/unireco_bird_example.jpg"
prompt = "你能分割出图中的鸟吗?,给出分割图"
output_text = inference(image_path, prompt)
print("output_text:", output_text)
