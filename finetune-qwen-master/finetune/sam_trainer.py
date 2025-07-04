#!/usr/bin/env python3
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import load_data
from .models.loss import DiceLoss, WeightedCrossEntropyWithLabelSmoothing
from .models.sam2_predictor import SAM2Predictor
from .plugin.metrics import compute_f1, compute_iou
from .utils.logger import ORIG_STDERR, ORIG_STDOUT


class DatasetHandler:
    """处理数据集加载和划分"""

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """处理数据批次合并"""
        collated = {}
        try:
            # 原来的合并逻辑…
            if "gt_img" in batch[0]:
                collated.update(
                    {
                        "gt_img_paths": [x["image_path"] for x in batch],
                        "gt_imgs": [x["gt_img"] for x in batch],
                        "gt_labels": torch.tensor(np.stack([x["gt_label"] for x in batch])),
                        "gt_masks": torch.tensor(np.stack([x["gt_mask"] for x in batch])),
                        "gt_boxes": [x["gt_boxes"] for x in batch],
                    }
                )
            return collated
        except Exception as e:
            logger.error("数据批次合并失败，请检查数据格式")
            traceback.print_exc()
            raise e

    @staticmethod
    def load_datasets(paths: List[str], seed: int, split_ratio: float, split_box_num=3) -> Tuple[Dataset, Dataset]:
        """加载并划分数据集"""
        if not 0.0 <= split_ratio < 1.0:
            raise ValueError("split_ratio must be in [0.0, 1.0)")

        entries = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"Dataset path {p} does not exist.")
            entries.append(
                {
                    "dataset_path": str(path.absolute()),
                    "pred_mask": True,
                    "resize_image": 1024,
                    "add_expert_feat": False,
                    "split_box_num": split_box_num,
                }
            )

        return load_data(entries, split_dataset_ratio=split_ratio, template=None, seed=seed)


class SAMTrainingConfig:
    """封装训练配置参数"""

    def __init__(
        self,
        dataset_paths: List[str],
        ds_config: str = "ds_config.json",
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        batch_size: int = 4,
        seed: int = 42,
        split_ratio: float = 0.1,
        epochs: int = 10,
        num_workers: int = 4,
        model_name: str = "facebook/sam2-hiera-tiny",
        hidden_size: int = 2048,
        patience: int = 5,
        debug: bool = False,
        resume_from: Optional[str] = None,
        gradient_accumulation_steps=2,
        split_box_num: int = 3,
    ):
        self.dataset_paths = dataset_paths
        self.ds_config = ds_config
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.seed = seed
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.num_workers = num_workers
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.patience = patience
        self.debug = debug
        self.resume_from = resume_from  # 新增断点恢复参数
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.split_box_num = split_box_num


class SAMTrainer:
    """管理模型训练全流程"""

    def __init__(
        self,
        engine: deepspeed.DeepSpeedEngine,
        model: SAM2Predictor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        save_dir: str = "./sam_checkpoints",
        patience: int = 3,  # 早停配置伦次
        loss_config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
        max_checkpoints=5,
    ):
        self.engine = engine
        self.local_rank = engine.local_rank
        self.world_size = engine.world_size
        self.device = device
        self.is_main = self.local_rank == 0

        # 只有主进程才创建 tqdm、logger 输出，非主进程重定向 stdout/stderr：
        if not self.is_main:
            sys.stdout = open(os.devnull, "w")

        self.engine.to(self.device)
        self.model = model
        self.model_name = model.model_name

        # 如果多卡训练，为 DataLoader 添加 DistributedSampler，避免数据重叠
        if self.world_size > 1:
            dataset = train_loader.dataset
            batch_size = train_loader.batch_size
            # 如果样本数 < batch_size，就不要丢弃最后一个不满 batch
            drop_last = len(dataset) >= batch_size

            # 训练集
            train_sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
            )
            self.train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=train_sampler,
                num_workers=0,
                collate_fn=train_loader.collate_fn,
                drop_last=drop_last,
            )
            # 验证集（不打乱）
            if val_loader is not None:
                val_sampler = DistributedSampler(
                    val_loader.dataset,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    shuffle=False,
                )
                self.val_loader = DataLoader(
                    val_loader.dataset,
                    batch_size=val_loader.batch_size,
                    sampler=val_sampler,
                    num_workers=val_loader.num_workers,
                    collate_fn=val_loader.collate_fn,
                    drop_last=False,
                )
            else:
                self.val_loader = None
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader

        self.base_save_dir = Path(save_dir)
        self.patience = patience
        self.loss_config = loss_config or {"dice_weight": 1.0, "bce_weight": 1.0, "cls_weight": 1.0}
        self.resume_from = resume_from
        self.training_history = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.start_epoch = 1
        self.max_checkpoints = max_checkpoints

        # 创建以时间戳命名的新文件夹用于保存当前训练的检查点和日志
        self.run_dir = self.base_save_dir
        self.save_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"

        # 创建目录
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 使用新的日志目录初始化TensorBoard
        if self.is_main:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # 记录当前运行的目录信息
        logger.info(f"创建新的训练运行目录: {self.run_dir}")
        logger.info(f"检查点将保存到: {self.save_dir}")
        logger.info(f"日志将保存到: {self.log_dir}")

        # 损失函数设计
        self.dice_loss = DiceLoss(use_sigmoid=True)
        self.loss_cls = WeightedCrossEntropyWithLabelSmoothing(
            pos_weight=1.0, loss_weight=self.loss_config["cls_weight"], use_sigmoid=True
        )

    def _save_checkpoint(self, epoch: int) -> None:
        """保存模型检查点并维护最新版本"""
        ckpt_tag = f"Epoch{epoch:03d}"
        # 使用 DeepSpeed 保存完整 checkpoint（包含 optimizer/zero 状态）
        # self.engine.save_checkpoint(str(self.save_dir), ckpt_tag)

        # 2) 仅主进程实际写文件与清理
        if not self.is_main:
            return

        # 2) 保存训练状态，以便断点恢复
        state = {
            "state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_no_improve": self.epochs_no_improve,
            "training_history": self.training_history,
        }
        state_file = self.save_dir / f"{ckpt_tag}.pt"
        torch.save(state, state_file)
        logger.info(f"Saved checkpoint: {ckpt_tag} in {self.run_dir}")

        # 仅由主进程清理旧 checkpoint 文件，只保留最新 self.max_checkpoints
        pattern = re.compile(r"epoch(\d+)")
        all_ckpts = sorted(
            [p for p in self.save_dir.iterdir() if p.is_dir() and pattern.search(p.name)],
            key=lambda x: int(pattern.search(x.name).group(1)),
            reverse=True,
        )
        for old_dir in all_ckpts[self.max_checkpoints :]:
            shutil.rmtree(old_dir, ignore_errors=True)
            # 同时删除对应的训练状态文件
            old_state = self.save_dir / f"{old_dir.name}.pt"
            if old_state.exists():
                old_state.unlink()

    def _load_checkpoint(self) -> int:
        """加载检查点和训练状态"""
        if not self.resume_from:
            return 1
        else:
            raise ValueError("还未实现 resume_from 的路径解析")

    def compute_loss(self, pred_masks, gt_masks, pred_labels, gt_labels):
        """计算混合损失：Dice Loss + BCE Loss"""
        # mask损失函数计算
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks) * self.loss_config["bce_weight"]
        dice_loss = self.dice_loss(pred_masks, gt_masks) * self.loss_config["dice_weight"]
        # labels损失函数计算
        cls_loss = self.loss_cls(pred_labels, gt_labels)
        loss = bce_loss + dice_loss + cls_loss
        return loss, bce_loss, dice_loss, cls_loss

    def _evaluate(self):
        """评估模型在验证集上的性能"""
        self.engine.eval()
        val_iou_sum, val_loss_sum, val_f1_sum = 0.0, 0.0, 0.0

        if self.is_main:
            val_pbar = tqdm(
                total=len(self.val_loader), desc="Validation", dynamic_ncols=True, leave=True, file=ORIG_STDOUT
            )
        else:
            val_pbar = None

        with torch.no_grad():
            for batch in self.val_loader:
                gt_masks = batch["gt_masks"].to(self.device)
                gt_labels = batch["gt_labels"].to(self.device)
                preds_masks, pred_labels = self.engine.module(batch["gt_imgs"], None, None, batch["gt_boxes"])
                loss, _, _, _ = self.compute_loss(preds_masks, gt_masks, pred_labels, gt_labels)
                iou = compute_iou(gt_masks, preds_masks)
                f1 = compute_f1(gt_masks, preds_masks)
                val_loss_sum += loss.item()
                val_iou_sum += iou.item()
                val_f1_sum += f1.item()
                if self.is_main:
                    val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}", "f1": f"{f1:.4f}"})
                    val_pbar.update(1)

        if self.is_main:
            val_pbar.close()

        # 分布式聚合：先转 tensor
        val_loss_tensor = torch.tensor(val_loss_sum, device=self.device) / len(self.val_loader)
        val_iou_tensor = torch.tensor(val_iou_sum, device=self.device) / len(self.val_loader)
        val_f1_tensor = torch.tensor(val_f1_sum, device=self.device) / len(self.val_loader)

        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_iou_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_f1_tensor, op=dist.ReduceOp.SUM)

        # 计算平均
        avg_val_loss = val_loss_tensor.item()
        avg_val_iou = val_iou_tensor.item()
        avg_val_f1 = val_f1_tensor.item()

        return avg_val_loss, avg_val_iou, avg_val_f1

    def train(self, epochs: int) -> None:
        """执行训练主循环，支持断点恢复"""
        # 尝试恢复检查点
        self.start_epoch = self._load_checkpoint()
        for epoch in range(self.start_epoch, epochs + 1):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            self.engine.train()
            train_loss = 0.0
            train_bce_loss = 0.0
            train_dice_loss = 0.0
            train_cls_loss = 0.0
            train_iou_sum = 0.0
            train_f1_sum = 0.0
            epoch_start_time = time.time()

            # 添加进度条
            train_pbar = (
                tqdm(
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch}/{epochs} [Train]",
                    dynamic_ncols=True,
                    leave=True,
                    file=ORIG_STDOUT,
                )
                if self.is_main
                else None
            )
            self.engine.train()

            for batch in self.train_loader:
                gt_masks = batch["gt_masks"].to(self.device)
                gt_labels = batch["gt_labels"].to(self.device)

                # 执行SAM训练
                preds_masks, pred_labels = self.engine.module(batch["gt_imgs"], None, None, batch["gt_boxes"])
                loss, bce_loss, dice_loss, cls_loss = self.compute_loss(preds_masks, gt_masks, pred_labels, gt_labels)
                iou = compute_iou(gt_masks, preds_masks, use_sigmoid=True)
                f1 = compute_f1(gt_masks, preds_masks, use_sigmoid=True)

                # 反向传播
                self.engine.backward(loss)
                self.engine.step()
                train_loss += loss
                train_iou_sum += iou
                train_f1_sum += f1
                train_bce_loss += bce_loss.item()
                train_dice_loss += dice_loss.item()
                train_cls_loss += cls_loss.item()

                # 更新进度条，显示当前损失
                if self.is_main:
                    train_pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "iou": f"{iou:.4f}",
                            "f1": f"{f1:.4f}",
                            "bce_loss": f"{bce_loss.item():.4f}",
                            "dice_loss": f"{dice_loss.item():.4f}",
                            "cls_loss": f"{cls_loss.item():.4f}",
                        }
                    )
                    train_pbar.update(1)

            if self.is_main:
                train_pbar.close()

            # 分布式聚合训练指标
            t_loss = torch.tensor(train_loss, device=self.device)
            t_bce = torch.tensor(train_bce_loss, device=self.device)
            t_cls = torch.tensor(train_cls_loss, device=self.device)
            t_dice = torch.tensor(train_dice_loss, device=self.device)
            t_iou = torch.tensor(train_iou_sum, device=self.device)
            t_f1 = torch.tensor(train_f1_sum, device=self.device)
            dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_iou, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_bce, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_dice, op=dist.ReduceOp.SUM)
            avg_train_loss = t_loss.item() / len(self.train_loader)
            avg_train_bce_loss = t_bce.item() / len(self.train_loader)
            avg_train_dice_loss = t_dice.item() / len(self.train_loader)
            avg_train_cls_loss = t_cls.item() / len(self.train_loader)
            avg_train_iou = t_iou.item() / len(self.train_loader)
            avg_train_f1 = t_f1.item() / len(self.train_loader)

            # 评估验证集
            if self.val_loader:
                val_loss, val_iou, val_f1 = self._evaluate()
            else:
                val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0

            epoch_time = time.time() - epoch_start_time

            # 记录训练历史
            self.training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "train_iou": avg_train_iou,
                    "train_f1": avg_train_f1,
                    "val_iou": val_iou,
                    "val_f1": val_f1,
                    "train_bce_loss": avg_train_bce_loss,
                    "train_dice_loss": avg_train_dice_loss,
                    "train_cls_loss": avg_train_cls_loss,
                    "epoch_time": epoch_time,
                }
            )

            # 记录日志
            # 同步到此处，确保各卡都完成计算
            dist.barrier()
            self._save_checkpoint(epoch)
            self._log_metrics(
                epoch,
                avg_train_loss,
                val_loss,
                avg_train_iou,
                avg_train_f1,
                val_iou,
                val_f1,
                avg_train_bce_loss,
                avg_train_dice_loss,
                avg_train_cls_loss,
                epoch_time,
            )
            dist.barrier()

            # 早停判断
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # 先更新本 rank 状态，再广播给所有 rank
            if self.patience > 0:
                stop = torch.tensor(int(self.epochs_no_improve >= self.patience), device=self.device)
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if stop.item() == 1:
                    if self.is_main:
                        logger.info("Early stopping triggered.")
                    break

        if self.is_main:
            self.writer.close()

    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_iou: float = None,
        train_f1: float = None,
        val_iou: float = None,
        val_f1: float = None,
        train_bce_loss: float = None,
        train_dice_loss: float = None,
        train_cls_loss: float = None,
        epoch_time: Optional[float] = None,
    ) -> None:
        """记录训练指标"""
        if not self.is_main:
            return

        # 构建日志消息
        iou_str = ""
        f1_str = ""
        if train_iou is not None:
            iou_str = f" | Train IoU: {train_iou:.4f}"
        if train_f1 is not None:
            f1_str = f" | Train F1: {train_f1:.4f}"
        if val_iou is not None and self.val_loader:
            iou_str += f" | Val IoU: {val_iou:.4f}"
        if val_f1 is not None and self.val_loader:
            f1_str += f" | Val F1: {val_f1:.4f}"

        loss_str = ""
        if train_bce_loss is not None:
            loss_str = f" | Train BCE: {train_bce_loss:.4f} | Train Dice: {train_dice_loss:.4f} | Train CLS: {train_cls_loss:.4f}"

        # 新增F1指标拼接

        time_str = f" | Time: {epoch_time:.2f}s" if epoch_time is not None else ""

        # 仅在主进程中记录日志
        logger.info(
            f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            f"{loss_str}{iou_str}{f1_str}{time_str}"
        )

        # 记录到TensorBoard
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        if self.val_loader:
            self.writer.add_scalar("Loss/Val", val_loss, epoch)

        # 记录独立损失
        if train_bce_loss is not None:
            self.writer.add_scalar("Loss/Train_BCE", train_bce_loss, epoch)
            self.writer.add_scalar("Loss/Train_Dice", train_dice_loss, epoch)
            self.writer.add_scalar("Loss/Train_CLS", train_cls_loss, epoch)

        # TensorBoard
        if train_f1 is not None:
            self.writer.add_scalar("F1/Train", train_f1, epoch)
        if val_f1 is not None and self.val_loader:
            self.writer.add_scalar("F1/Val", val_f1, epoch)
        if train_iou is not None:
            self.writer.add_scalar("IoU/Train", train_iou, epoch)
        if val_iou is not None and self.val_loader:
            self.writer.add_scalar("IoU/Val", val_iou, epoch)

        if epoch_time is not None:
            self.writer.add_scalar("Performance/EpochTime", epoch_time, epoch)
