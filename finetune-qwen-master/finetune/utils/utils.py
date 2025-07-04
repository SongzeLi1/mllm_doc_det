# 获取当前可用的GPU数量
import os


def get_available_gpu_count():

    # 方法2: 使用CUDA_VISIBLE_DEVICES环境变量
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        # 方法1: 使用PyTorch
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
