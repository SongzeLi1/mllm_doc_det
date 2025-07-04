#!/usr/bin/env python3
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed import init_inference
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.sam2_predictor import SAM2Predictor
from .plugin.metrics import compute_auc, compute_label_metrics, compute_pixel_metrics
from .utils.logger import ORIG_STDOUT


class SAMInference:
    """
    推理类: 加载训练好的 SAM2Predictor，执行推理并计算 IoU。
    支持传入 checkpoint 目录，遍历目录下所有文件做推理并分别记录日志。
    """

    def __init__(
        self,
        model: SAM2Predictor,
        device: Optional[torch.device] = None,
        threshold=0.5,
        rank=0,
        world_size=1,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.threshold = threshold
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.world_size = world_size
        self.rank = rank
        self.use_deepspeed = self.world_size > 1

    def _load_state(self, checkpoint_data):
        """按 ckpt_path 重载 model 的参数。"""
        init_args = dict(
            model=self.model,
            tensor_parallel={"tp_size": self.world_size},
            dtype=torch.float32,
            replace_with_kernel_inject=True,
        )

        # DeepSpeed 初始化
        if self.use_deepspeed:
            self.model = init_inference(**init_args)
            self.model.module.load_state_dict(checkpoint_data)
        else:
            # 加载权重
            self.model.load_state_dict(checkpoint_data)
        self.model = self.model.eval()

    def gather_tensor(self, tensor):
        # 先确保是1维
        tensor = tensor.view(-1)
        out_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(out_list, tensor)
        return torch.cat(out_list, dim=0)

    def _infer(
        self,
        dataloader: DataLoader,
        save_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        rank = self.rank
        save_path = Path(save_dir) if save_dir else None
        if rank == 0 and save_path:
            save_path.mkdir(parents=True, exist_ok=True)
        # 新增：分子分母累加器
        metrics_sum = defaultdict(list)
        self.model.eval()
        logger.info("Start inference...")
        pbar = tqdm(
            dataloader,
            desc="Inferring",
            dynamic_ncols=True,
            disable=(not show_progress) or (rank != 0),
            file=ORIG_STDOUT,
        )
        with torch.inference_mode():
            for idx, batch in enumerate(pbar):
                gt_masks: torch.Tensor = batch["gt_masks"].to(self.device)
                gt_labels: torch.Tensor = batch["gt_labels"].to(self.device)
                img_paths = batch["gt_img_paths"]
                pred_masks, pred_labels = self.model(batch["gt_imgs"], None, None, batch.get("gt_boxes", None))
                pred_masks = torch.sigmoid(pred_masks)

                # 假设 compute_pixel_metrics 返回的是每个batch的均值
                seg_res = compute_pixel_metrics(
                    gt_masks.detach().cpu().numpy(),
                    pred_masks.detach().cpu().numpy(),
                    self.threshold,
                    use_sigmoid=False,
                )
                # seg_res: dict, 每个 key 是 [N,] shape
                for k, v in seg_res.items():
                    metrics_sum[k].append(v)

                if rank == 0 and save_dir:
                    for i in range(pred_masks.shape[0]):
                        mask = pred_masks[i].cpu().numpy()
                        mask = (mask > self.threshold).astype(np.uint8) * 255
                        img_name = Path(img_paths[i]).name
                        mask_path = save_path / img_name
                        cv2.imwrite(str(mask_path), mask)

                if rank == 0:
                    # 只显示当前 batch 的均值
                    pbar.set_postfix(
                        {
                            "IoU": f"{np.mean(seg_res['seg_iou']):.4f}",
                            "F1-bin": f"{np.mean(seg_res['seg_f1_binary']):.4f}",
                            "Acc": f"{np.mean(seg_res['seg_acc']):.4f}",
                        }
                    )

        # 分布式聚合
        if self.use_deepspeed:
            for k in metrics_sum.keys():
                # 先把本地所有 batch 的均值合成总和与总数
                local_sum = torch.tensor(float(np.sum(metrics_sum[k])), device=self.device)
                local_count = torch.tensor(len(metrics_sum[k]), device=self.device)
                # 全局求和
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
                # 用全局总和和总数算全局均值
                metrics_sum[k] = (local_sum.item(), local_count.item())

        # 统一计算均值
        local_metrics = {}
        for k in metrics_sum.keys():
            if self.use_deepspeed:
                total_sum, total_count = metrics_sum[k]
                local_metrics[k] = total_sum / total_count if total_count > 0 else 0.0
            else:
                local_metrics[k] = sum(metrics_sum[k]) / len(metrics_sum[k])
        return local_metrics

    def infer(
        self,
        checkpoint_data,
        dataloader: DataLoader,
        save_dir: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        支持 checkpoint 目录或单文件：
        - 如果是目录，遍历所有文件，分别做推理并记录日志和结果。
        - 否则直接对单个 checkpoint 推理。
        """
        # 单 checkpoint 推理
        try:
            self._load_state(checkpoint_data)
            logger.info(f"Loaded checkpoint from {checkpoint_data.get('model_name')}")
            res = self._infer(dataloader, save_dir, show_progress)
            if self.rank == 0:
                logger.info(
                    f"Results for {checkpoint_data.get("model_name")} | "
                    + " | ".join(f"{k}: {v:.4f}" for k, v in res.items())
                )
            return res
        finally:
            # 清理分布式环境
            if self.use_deepspeed:
                dist.barrier()
                dist.destroy_process_group()
