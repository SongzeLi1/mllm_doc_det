import concurrent.futures
import math

import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..datasets import load_data
from ..utils.logger import ORIG_STDOUT
from .custom_request import CustomRequest


class InferRequestLoader:
    @staticmethod
    def distribute(
        dataset: Dataset,
        num_workers=4,
        batch_size=16,
        pin_memory=True,
        distributed=True,
    ) -> DataLoader:
        """多进程多线程迭代访问数据集

        Args:
            dataset: 待处理的数据集
            batch_size: 批处理大小
            num_workers: 数据加载的工作进程数
            pin_memory: 是否将数据固定到CUDA内存
            persistent_workers: 是否保持工作进程存活
            distributed: 是否使用分布式训练
            rank: 当前进程的rank
            world_size: 总进程数
            load_into_memory: 是否将所有数据加载到内存中返回list

        Returns:
            如果load_into_memory=True，返回list；否则返回DataLoader
        """
        # 如果使用分布式训练，添加DistributedSampler
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset=dataset, shuffle=False)

        # 创建DataLoader进行多进程数据加载
        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size=batch_size,
            collate_fn=InferRequestLoader.collate_fn,
            sampler=sampler,
        )

        # 计算批次数（若dataset实现了__len__则根据dataset长度计算，否则使用dataloader的长度）
        total_batches = math.ceil(len(dataset) / batch_size) if hasattr(dataset, "__len__") else len(dataloader)
        total_samples = len(dataset) if hasattr(dataset, "__len__") else "N/A"
        logger.info(f"数据集批次数: {total_batches}")
        logger.info(f"数据集每批次数量: {batch_size}")
        logger.info(f"数据集总数量: {total_samples}")

        return dataloader

    @staticmethod
    def collate_fn(batch, task_type="causal_lm"):
        for data in batch:
            if task_type == "causal_lm":
                CustomRequest.remove_response(data["template_inputs"].messages)
            else:
                data.pop("label", None)
        return batch

    @staticmethod
    def process_dataset(dataloader, max_workers=32, task_type="causal_lm"):
        """使用线程池并行处理DataLoader中的批次数据

        Args:
            dataloader: 数据加载器（需支持索引访问）
            max_workers: 最大线程数

        Returns:
            处理后的数据列表
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(lambda i: dataloader[i], i) for i in range(len(dataloader))]
            # 使用线程池执行器并返回迭代器
            # 并行处理并逐个产出结果
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if task_type == "causal_lm":
                    CustomRequest.remove_response(data["template_inputs"].messages)
                else:
                    data.pop("label", None)
                yield data

    @staticmethod
    def process_sample(sample, system_msg=None):
        """处理单个样本"""
        system_msg = system_msg or []
        return CustomRequest(
            images=sample.get("images"),
            messages=system_msg + sample.get("messages", [])[:1],
            gt_label=sample["gt_label"],
            gt_mask=sample.get("gt_mask"),
        )

    @staticmethod
    def load(dataset=None, system_msg=None, template=None, seed=None, dataset_paths=None):
        """加载数据集"""
        system_msg = system_msg or []
        if dataset is None:
            dataset, _ = load_data(dataset_paths, template, seed=seed)

        # 使用ThreadPoolExecutor并行处理样本
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = list(
                tqdm(
                    executor.map(lambda sample: InferRequestLoader.process_sample(sample, system_msg), dataset),
                    total=len(dataset),
                    desc="加载推理数据集",
                    ncols=100,
                    file=ORIG_STDOUT,
                )
            )
        return futures
