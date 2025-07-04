from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import ConcatDataset, Dataset, random_split

from .contrastive_dataset import ContrastiveDataset
from .controllable_dataset import ControllableDataset
from .custom_dataset import CustomDataset


class DatasetType(Enum):
    contrastive = ContrastiveDataset
    controllable = ControllableDataset
    custom = CustomDataset


def load_data(
    train_args_list: List[Dict],
    template,
    dataset_type: Union[str, DatasetType] = None,
    split_dataset_ratio: float = 0.0,
    seed: Union[int, np.random.RandomState, None] = None,
    **kwargs,
) -> Tuple[Dataset, Optional[Dataset]]:
    # 载入数据集
    train_datasets = []
    val_datasets = []

    # 每个数据集取一部分作为验证集
    if isinstance(seed, int):
        generator = torch.Generator().manual_seed(seed)
    elif isinstance(seed, np.random.RandomState):
        generator = torch.Generator().manual_seed(seed.randint(0, 2**32 - 1))
    elif seed is None:
        generator = torch.Generator().manual_seed(torch.seed())
    else:
        raise ValueError("seed must be int or np.random.RandomState")

    # 数据集后处理
    for data_args in train_args_list:
        logger.info(f"Loading dataset: {data_args}")
        # 公共初始化参数
        init_kwargs = {"encode_func": template, "random_state": seed}

        # 如果用户没指定 dataset_type，则依次尝试 custom → controllable → contrastive
        if dataset_type is None:
            for dt in (DatasetType.custom, DatasetType.controllable, DatasetType.contrastive):
                DatasetClass = dt.value
                if isinstance(data_args, list):
                    train_dataset: CustomDataset = DatasetClass(*data_args, **init_kwargs)
                elif isinstance(data_args, dict):
                    train_dataset: CustomDataset = DatasetClass(**data_args, **init_kwargs)
                else:
                    train_dataset: CustomDataset = DatasetClass(data_args, **init_kwargs)
                if train_dataset.check_data():
                    logger.info(f"Successfully loaded dataset with type: {dt}")
                    break
                dataset_type = dt  # 记录成功的类型
            else:
                raise ValueError(f"无法为 data_args 推断出 dataset_type: {data_args}")
        else:
            # 确保 dataset_type 为 DatasetType
            DatasetClass = dataset_type.value if isinstance(dataset_type, DatasetType) else DatasetType[dataset_type].value
            if isinstance(data_args, list):
                train_dataset = DatasetClass(*data_args, **init_kwargs)
            elif isinstance(data_args, dict):
                train_dataset = DatasetClass(**data_args, **init_kwargs)
            else:
                train_dataset = DatasetClass(data_args, **init_kwargs)

        train_dataset.print_info()
        if split_dataset_ratio > 0:
            train_size = int(len(train_dataset) * (1 - split_dataset_ratio))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
            val_datasets.append(val_dataset)
        train_datasets.append(train_dataset)

    # 合并数据集
    train_datasets = ConcatDataset(train_datasets)
    val_datasets = ConcatDataset(val_datasets) if val_datasets else None

    shuffled_indices = torch.randperm(len(train_datasets), generator=generator)
    train_datasets = torch.utils.data.Subset(train_datasets, shuffled_indices.tolist())

    if val_datasets is not None:
        shuffled_indices = torch.randperm(len(val_datasets), generator=generator)
        val_datasets = torch.utils.data.Subset(val_datasets, shuffled_indices.tolist())

    return train_datasets, val_datasets
