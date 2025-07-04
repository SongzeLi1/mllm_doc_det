import concurrent.futures
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finetune_sam import DatasetHandler


def process_item(idx, dataset):
    """处理数据集中单条数据的函数"""
    try:
        return dataset[idx]
    except Exception as e:
        print(f"处理数据项 {idx} 时出错: {str(e)}")
        return None


def main():

    train_datasets, val_datasets = DatasetHandler.load_datasets(
        [
            "/home/yuyangxin/data/finetune-qwen/resource/datasets/without_instruct/MIML_Part1_fake.json",
        ],
        42,
        0,
    )

    # 多线程遍历数据集
    length = len(train_datasets)
    max_workers = min(32, os.cpu_count() * 2)  # 根据CPU核心数动态设置线程数

    print(f"开始使用 {max_workers} 个线程处理 {length} 条数据...")

    print(f"开始单线程处理 {length} 条数据...")
    for idx in range(length):
        try:
            process_item(idx, train_datasets)
        # 这里可以对处理结果进行进一步操作
        except Exception:
            continue

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # 创建所有任务
    #     futures = {executor.submit(process_item, l, train_datasets): l for l in range(length)}

    #     # 不使用tqdm显示进度
    #     for future in concurrent.futures.as_completed(futures):
    #         idx = futures[future]
    #         try:
    #             result = future.result()
    #             # 这里可以对处理结果进行进一步操作
    #             # 如果需要收集结果，可以添加到一个列表中
    #         except Exception as e:
    #             continue


if __name__ == "__main__":
    main()
