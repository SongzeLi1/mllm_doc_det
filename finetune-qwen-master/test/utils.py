import os

import torch


# 选择目前可用的GPU, GPU的可用内存要超过20G
def get_available_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("没有找到可用的GPU设备")

    available_gpus = []
    for i in range(torch.cuda.device_count()):
        # 使用 mem_get_info 获取实际可用内存
        free_memory, total_memory = torch.cuda.mem_get_info(i)
        free_memory_gb = free_memory / (1024**3)
        if free_memory_gb > 20:
            available_gpus.append((i, free_memory_gb))

    if available_gpus:
        selected_gpu = available_gpus[0][0]
        print(f"选择GPU {selected_gpu}，可用内存: {available_gpus[0][1]:.2f}GB")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        return selected_gpu
    raise RuntimeError("没有找到可用内存超过20GB的GPU")
