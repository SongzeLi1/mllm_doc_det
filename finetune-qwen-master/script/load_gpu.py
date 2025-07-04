import argparse
import os
import re
import subprocess
import sys

import numpy as np


def get_gpu_memory():
    """获取GPU剩余内存(GB)，支持Linux和Windows系统"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        return [int(x) for x in result.stdout.strip().split("\n")]
    except Exception as e:
        raise RuntimeError(f"获取GPU信息失败: {str(e)}") from e


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--memory", type=int, default=0, help="要求GPU的剩余显存最小值(GB)")
    parser.add_argument("--retain", type=int, default=0, help="需要保留的GPU数量")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    parser.add_argument("--help", action="store_true", help="显示帮助信息")
    return parser.parse_args()


def load_gpu():
    args = parse_arguments()

    if args.help:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} [--memory MIN_MEMORY_GB] [--retain RETAIN_GPU] [--verbose]")
        print("--memory\t要求GPU的剩余显存最小值(GB)")
        print("--retain\t需要保留的GPU数量(默认为0)")
        print("--verbose\t显示详细信息")
        print("--help\t\t显示此帮助信息")
        return

    try:
        # 将输入参数的内存单位由GB转为MB
        required_memory_mb = args.memory * 1024

        # 获取所有GPU的剩余内存(MB)
        free_memories = get_gpu_memory()
        gpu_count = len(free_memories)

        if gpu_count == 0:
            raise ValueError("未检测到任何GPU设备")

        # 筛选符合条件的GPU
        valid_gpus = [(i, mem) for i, mem in enumerate(free_memories) if mem >= required_memory_mb]
        if not valid_gpus:
            raise ValueError(f"没有找到剩余内存大于{args.memory}GB的GPU")

        # 处理retain参数
        retain_count = max(0, args.retain)
        if retain_count > len(valid_gpus):
            raise ValueError(f"保留GPU数量{retain_count}超过了可用GPU数量{len(valid_gpus)}")

        # 按照剩余内存从大到小排序
        sorted_gpus = sorted(valid_gpus, key=lambda x: x[1], reverse=True)
        selected = sorted_gpus[:-retain_count] if retain_count > 0 else sorted_gpus
        selected_ids = [str(gpu[0]) for gpu in selected]

        # 设置环境变量
        cuda_visible_devices = ",".join(selected_ids)
        # 详细信息输出
        if args.verbose:
            print(f"检测到{gpu_count}个GPU:")
            for i, mem in enumerate(free_memories):
                status = "✓" if i in [g[0] for g in selected] else "✗"
                mem_gb = mem / 1024
                print(f"GPU {i}: {mem_gb}GB {status}")
            print(f"最终选择GPU: {', '.join(selected_ids)}")

        # 仅输出选择的GPU ID，方便脚本调用
        print(cuda_visible_devices)
        return cuda_visible_devices
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    load_gpu()
