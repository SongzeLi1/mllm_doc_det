#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from finetune.models.sam2_predictor import SAM2Predictor
from finetune.sam_infer import SAMInference
from finetune.sam_trainer import DatasetHandler
from finetune.utils import LoggerConfig


def configure_loader(
    ds,
    batch_size: int,
    num_workers: int,
    debug: bool,
    distributed: bool = False,  # 新增参数
    rank: int = 0,  # 新增参数
    world_size: int = 1,  # 新增参数
):
    train_ds, val_ds = DatasetHandler.load_datasets(ds, seed=42, split_ratio=0)
    target_ds = val_ds if val_ds is not None else train_ds
    if distributed:
        sampler = DistributedSampler(
            target_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    loader_args = {
        "batch_size": batch_size,
        "collate_fn": DatasetHandler.collate_fn,
        "num_workers": num_workers if not debug else 0,
        "pin_memory": not debug,
        "persistent_workers": not debug,
        "shuffle": False if sampler is not None else False,
        "sampler": sampler,
    }
    # DataLoader 不允许同时设置 shuffle=True 和 sampler!=None
    if sampler is not None:
        loader_args.pop("shuffle")
    return DataLoader(target_ds, **loader_args)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer SAM2Predictor on a dataset")
    parser.add_argument("--dataset_paths", nargs="+", required=True, help="数据集路径列表")
    parser.add_argument("--checkpoint", required=True, help="单个 checkpoint 文件或 checkpoint 目录")
    parser.add_argument("--save_dir", default=None, help="推理保存结果目录")
    parser.add_argument("--save_mask", action="store_true", default=False, help="推理时是否保存mask")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--debug", action="store_true", help="调试模式，禁用多进程数据加载")
    parser.add_argument("--use_deepspeed", action="store_true", default=True, help="启用 DeepSpeed 加速推理")
    args = parser.parse_args()

    # 参数检测
    assert args.dataset_paths, "数据集路径不能为空"
    assert args.checkpoint, "checkpoint 路径不能为空"
    return args


def main(args, distributed, rank, world_size):

    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"checkpoint 路径不存在: {args.checkpoint}"

    # 日志与输出目录
    timestamp = str(time.strftime("%Y%m%d_%H%M%S"))[:-2]
    if args.save_dir is None:
        # 在 checkpoint 所在目录下以模型名称命名文件夹
        checkpoint_path = Path(args.checkpoint)
        checkpoint_name = f"{checkpoint_path.stem.split('-')[0]}"
        args.save_dir = checkpoint_path.parent.parent / checkpoint_name
    LoggerConfig.configure(args.save_dir, timestamp, "inference")
    logger.info(f"推理结果将保存至: {args.save_dir}")

    # 随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据加载器
    dataloader = configure_loader(
        ds=args.dataset_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        debug=args.debug,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    # 合并结果
    dataset_name = []
    for dataset_path in args.dataset_paths:
        dataset_name.append(str(Path(dataset_path).stem))
    dataset_name = "_".join(dataset_name)

    # 模型与推理器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint_data.get("model_name", "facebook/sam2-hiera-base-plus")
    model = SAM2Predictor(model_name=model_name, hidden_size=args.hidden_size, is_finetune=True)
    inference = SAMInference(model=model, device=device, threshold=args.threshold, rank=rank, world_size=world_size)

    logger.info(f"模型名称: {model_name}")
    logger.info(f"推理数据集: {dataset_name}")
    logger.info(f"推理数据集路径: {args.dataset_paths}")
    logger.info(f"推理结果保存目录: {args.save_dir}")
    logger.info(f"使用 DeepSpeed 加速推理: {args.use_deepspeed}")
    if args.use_deepspeed:
        logger.info(f"DeepSpeed tensor 模型并行度: {inference.world_size}")

    # 执行推理
    if args.save_mask is True:
        mask_path = str(args.save_dir / f"{dataset_name}_masks")
    else:
        mask_path = None

    new_results = inference.infer(
        checkpoint_data,
        dataloader=dataloader,
        save_dir=mask_path,
        show_progress=True,
    )

    # 仅在rank为0时, 进行保存, 并汇总保存结果
    if inference.rank == 0:
        result_save_path = args.save_dir / "sam_results.json"
        # 读取内容
        if result_save_path.exists():
            try:
                with open(result_save_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except:
                results = {}
        else:
            results = {}

        # 合并结果
        results[dataset_name] = new_results
        with open(result_save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"推理完成，结果已保存到 {result_save_path}")


def init_distributed():
    """自动初始化分布式环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        distributed = True
    else:
        rank = 0
        world_size = 1
        distributed = False
    return distributed, rank, world_size


if __name__ == "__main__":
    args = parse_args()
    distributed, rank, world_size = init_distributed()
    main(args, distributed, rank, world_size)
