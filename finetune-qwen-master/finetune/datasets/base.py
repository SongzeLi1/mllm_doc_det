import json
import os
from enum import Enum
from typing import Dict, List

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path: str) -> Image.Image:
    """PIL image loader

    Args:
        path (str): image path

    Returns:
        Image.Image: PIL image (after np.array(x) becomes [0,255] int8)
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class TokenPosEnum(Enum):
    FRONT = 0
    BACK = 1


class BaseDataset(Dataset):
    def __init__(self, json_path, tokenizer, data_args):
        # 加载数据
        if isinstance(json_path, list):
            data = []
            for path in json_path:
                with open(path) as f:
                    cur_data = json.load(f)
                data.extend(cur_data)
        else:
            with open(json_path) as f:
                data = json.load(f)

        self.data = data
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.data)

    @staticmethod
    def convert_to_np(image: Image.Image, resolution: int) -> np.ndarray:
        """Convert an image to a NumPy array suitable for model input."""
        image = image.convert("RGB")
        image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
        return np.array(image).transpose(2, 0, 1)

    @staticmethod
    def load_image(image_path: str, mode="RGB") -> Image.Image:
        """Load an image from the given path."""
        # 判断文件存不存在
        image_path = os.path.expanduser(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert(mode)

    @staticmethod
    def load_conversation_templates(template_path: str) -> List[Dict[str, str]]:
        """Load conversation templates from a file."""
        templates = []
        with open(template_path, "r") as file:
            current_template = {}
            for line in file:
                line = line.strip()
                if line.startswith("user: "):
                    if current_template:
                        templates.append(current_template)
                        current_template = {}
                    current_template["user"] = line[len("user: ") :]
                elif line.startswith("assistant: "):
                    current_template["assistant"] = line[len("assistant: ") :]
            if current_template:
                templates.append(current_template)
        return templates
