import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finetune.models.experts import NPRModel


# 尝试载入模型
def test_load_npr():
    # 这里的路径需要根据实际情况修改
    model_path = Path("/pubdata/yuyangxin/swift-demo/finetune/models/experts/NPR/NPR.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 创建模型实例
    model = NPRModel()

    # 加载模型参数
    target_module = torch.load(model_path)["model"]
    model.load_state_dict(target_module, strict=True)


if __name__ == "__main__":
    test_load_npr()
    print("模型加载成功")
