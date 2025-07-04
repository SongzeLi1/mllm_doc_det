import argparse
import json
import socket
import time
from pathlib import Path
from typing import Optional, Tuple

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from finetune.models import SAM2Predictor
from finetune.sam_trainer import DatasetHandler, SAMTrainer, SAMTrainingConfig
from finetune.utils import LoggerConfig


def configure_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    config: SAMTrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """配置数据加载器"""
    loader_args = {
        "batch_size": config.batch_size,
        "collate_fn": DatasetHandler.collate_fn,
        "num_workers": config.num_workers if not config.debug else 4,
        "pin_memory": not config.debug,
        "persistent_workers": not config.debug,
        "prefetch_factor": 2 * max(1, config.num_workers),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args) if val_ds else None

    logger.info(f"是否开启DEBUG摸索模式: {config.debug}")
    logger.info(f"训练数据集大小: {len(train_ds)}")
    logger.info(f"训练数据加载器大小: {len(train_loader)}")
    logger.info(f"训练数据加载器批大小: {loader_args['batch_size']}")
    logger.info(f"训练数据加载器预取因子: {loader_args['prefetch_factor']}")
    logger.info(f"训练数据加载器工作线程数: {loader_args['num_workers']}")
    return train_loader, val_loader


def main(config: SAMTrainingConfig) -> None:
    dataset_name = []
    for dataset_path in args.dataset_paths:
        dataset_name.append(str(Path(dataset_path).stem))
    dataset_name = "_".join(dataset_name)

    # 实例化日志记录器
    timestamp = str(time.strftime("%Y%m%d_%H%M%S"))[:-2]
    model_name = config.model_name.split("/")[-1]
    config.save_dir = Path(config.save_dir) / f"{model_name}_{config.split_box_num}box_{dataset_name}_{timestamp}"

    # 判断config.save_dir是否是相对路径, 如果是相对路径, 则以当前文件路径作为基准路径
    if not config.save_dir.is_absolute():
        config.save_dir = Path(__file__).parent / config.save_dir
    LoggerConfig.configure(config.save_dir, timestamp)

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 创建模型
    model = SAM2Predictor(model_name=config.model_name, hidden_size=config.hidden_size, is_finetune=True)

    # 初始化DeepSpeed
    with open(config.ds_config) as f:
        ds_config = json.load(f)

    # 自动设置 DeepSpeed 批大小参数
    grad_steps = config.gradient_accumulation_steps
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    ds_config["train_micro_batch_size_per_gpu"] = config.batch_size
    ds_config["gradient_accumulation_steps"] = grad_steps
    ds_config["train_batch_size"] = config.batch_size * grad_steps * num_gpus
    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    torch.cuda.manual_seed_all(config.seed)

    # 加载数据集
    train_ds, val_ds = DatasetHandler.load_datasets(
        config.dataset_paths,
        config.seed,
        config.split_ratio,
        config.split_box_num,
    )
    train_loader, val_loader = configure_loaders(train_ds, val_ds, config)

    # 创建训练器
    trainer = SAMTrainer(
        engine=engine,
        model=model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=config.save_dir,
        patience=config.patience,
        resume_from=config.resume_from,
    )

    # 加载检查点
    if config.resume_from:
        logger.info(f"从检查点 '{config.resume_from}' 恢复训练")
    logger.info(f"开始训练，检查点将保存至: {config.save_dir}")
    trainer.train(config.epochs)
    logger.info(f"训练完成，检查点目录: {config.save_dir}")


def find_available_port() -> int:
    """找到一个未被占用的本地端口"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM2Predictor with DeepSpeed")
    parser.add_argument("--dataset_paths", nargs="+", required=True)
    parser.add_argument("--ds_config", default="ds_config.json")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.1)
    parser.add_argument("--split_box_num", type=int, default=3, help="切割的box数量")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", default="facebook/sam2-hiera-tiny")
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume_from", default=None, help="从检查点恢复训练，'latest'表示使用最新检查点")
    parser.add_argument("--debug", action="store_true", default=False, help="调试模式，减少数据加载器的工作线程数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")

    args = parser.parse_args()
    config = SAMTrainingConfig(**vars(args))
    try:
        main(config)
    finally:
        # 如果在分布式训练中, 确保关闭所有进程
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
