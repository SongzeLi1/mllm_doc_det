import concurrent.futures
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


def compute_iou(gt_masks, pred_masks, threshold: float = 0.5, eps: float = 1e-8, use_sigmoid=True):
    """
    计算 IoU 并取 batch 平均。
        gt_masks当中大于阈值的为True表示负样本,需要检测的区域,为False表示正样本不需要检测的区域?
        也就是要检测的 负样本/所有负样本待检测的区域
    Args:
        gt_masks: 真实 mask，形状为 (batch_size, 1, H, W) 或 (batch_size, H, W)
        pred_masks: 预测 mask，形状为 (batch_size, 1, H, W) 或 (batch_size, H, W)
        threshold: 二值化阈值
        eps: 避免除0错误
    """
    pred_bin = _binarize_mask(pred_masks, threshold, use_sigmoid=use_sigmoid)
    gt_bin = _binarize_mask(gt_masks, threshold, use_sigmoid=use_sigmoid)

    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"Shape mismatch: {pred_bin.shape} vs {gt_bin.shape}")

    # sum over H,W
    # sum over H,W
    if isinstance(pred_bin, np.ndarray):
        intersection = (pred_bin & gt_bin).sum(axis=(1, 2)).astype(np.float32)
        union = (pred_bin | gt_bin).sum(axis=(1, 2)).astype(np.float32)
        intersection = torch.from_numpy(intersection)
        union = torch.from_numpy(union)
    else:
        intersection = (pred_bin & gt_bin).sum(dim=(1, 2)).float()
        union = (pred_bin | gt_bin).sum(dim=(1, 2)).float()
    union = union.clamp_min(eps)
    return (intersection / union).mean()


def compute_f1(gt_masks, pred_masks, threshold: float = 0.5, eps: float = 1e-8, use_sigmoid=True):
    """
    计算 batch 上的平均 F1.
    """
    pred_bin = _binarize_mask(pred_masks, threshold, use_sigmoid)
    gt_bin = _binarize_mask(gt_masks, threshold, use_sigmoid)

    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"Shape mismatch: {pred_bin.shape} vs {gt_bin.shape}")

    # 转为 numpy，逐样本计算 F1
    if isinstance(pred_bin, torch.Tensor):
        pred_bin = pred_bin.cpu().numpy()
    if isinstance(gt_bin, torch.Tensor):
        gt_bin = gt_bin.cpu().numpy()

    batch_size = pred_bin.shape[0]
    f1s = []
    for i in range(batch_size):
        y_true = gt_bin[i].reshape(-1).astype(np.uint8)
        y_pred = pred_bin[i].reshape(-1).astype(np.uint8)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    return float(np.mean(f1s))


def compute_label_metrics(gt_labels, pred_probs, use_sigmoid=False) -> Dict[str, float]:
    # 是否对 pred_labels 做 sigmoid 处理
    if use_sigmoid:
        pred_probs = torch.sigmoid(pred_probs)
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.detach().cpu().numpy()
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.detach().cpu().numpy()

    # 输入展开
    if np.isscalar(gt_labels) or np.size(gt_labels) == 1:
        gt_labels = np.array([gt_labels])
    if np.isscalar(pred_probs) or np.size(pred_probs) == 1:
        pred_probs = np.array([pred_probs])

    # 过滤只含 0/1 的有效样本
    # 将其按照阈值0.5二值化
    y_true = gt_labels
    y_pred = (pred_probs > 0.5).astype(float)

    # 调用 sklearn 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # 将所有结果保留4位小数，未处理样本作为整数返回
    det_metrics = {
        "det_real_label": int(np.sum(y_true == 0)),
        "det_fake_label": int(np.sum(y_true == 1)),
        "det_tp": tp,
        "det_fp": fp,
        "det_fn": fn,
        "det_tn": tn,
        "det_accuracy": accuracy,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
    }
    # det_metrics = {
    #     k: f"{round(float(v), 4):.4f}" if k != "det_uncounted_samples" else f"{int(v)}" for k, v in det_metrics.items()
    # }
    return det_metrics


def compute_auc(gt, pred_probs, use_sigmoid=False) -> float:
    # 是否对 pred_labels 做 sigmoid 处理
    if use_sigmoid:
        pred_probs = torch.sigmoid(pred_probs)
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.detach().cpu().numpy()

    # 输入展开
    if np.isscalar(gt) or np.size(gt) == 1:
        gt = np.array([gt])
    if np.isscalar(pred_probs) or np.size(pred_probs) == 1:
        pred_probs = np.array([pred_probs])

    # 计算AUC
    try:
        if len(np.unique(gt)) < 2:
            auc = 0.0  # 只有一个类别，AUC无意义
        else:
            auc = roc_auc_score(gt, pred_probs)
    except Exception:
        auc = 0.0
    return auc


def _binarize_mask(x, threshold: float = 0.5, use_sigmoid: bool = False):
    """统一的 sigmoid+threshold，并 squeeze 通道维。"""
    # 1. 类型、设备、维度处理
    if use_sigmoid:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = torch.sigmoid(x)
        x = x.detach().cpu().numpy()
    else:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            # 判断类型, 如果都为bool类型, 则直接返回
            if x.dtype == np.bool_:
                return x
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    if x.min() < 0 or x.max() > 1:
        raise ValueError("x should be in [0, 1] range.")

    # squeeze 单通道
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    # 3. binarize
    return x >= threshold


def compute_pixel_metrics(gt_masks, pred_masks, threshold: float = 0.5, use_sigmoid: bool = True) -> Dict[str, float]:
    """
    任务: 进行图像真伪mask检测 - 样本平均
    要求: 计算像素级指标，包括准确率、IoU、精确率、召回率和F1分数。
    GT Mask是由的数据格式如下, 像素点值为0或1:
        0表示未被编辑过修改的区域
        1表示被修改编辑过的区域
    """
    # 1. 转 numpy 并 binarize
    if use_sigmoid is True:
        gt_bins = _binarize_mask(gt_masks, threshold, use_sigmoid=use_sigmoid)
        pred_bins = _binarize_mask(pred_masks, threshold, use_sigmoid=use_sigmoid)
        if pred_bins.shape != gt_bins.shape:
            raise ValueError(f"Shape mismatch: {pred_bins.shape} vs {gt_bins.shape}")
        batch_size = gt_bins.shape[0]
    else:
        gt_bins = gt_masks
        pred_bins = pred_masks
        batch_size = len(gt_bins)

    result = defaultdict(list)

    def process_one(i):
        y_true = np.array(gt_bins[i]).reshape(-1).astype(np.uint8)
        y_pred = np.array(pred_bins[i]).reshape(-1).astype(np.uint8)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0, average="binary")
        recall = recall_score(y_true, y_pred, zero_division=0, average="binary")
        f1_binary = f1_score(y_true, y_pred, zero_division=0, average="binary")
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        iou = jaccard_score(y_true, y_pred, zero_division=0)
        return {
            "seg_acc": acc,
            "seg_precision": precision,
            "seg_recall": recall,
            "seg_f1_binary": f1_binary,
            "seg_iou": iou,
            "seg_f1_macro": f1_macro,
            "seg_f1_micro": f1_micro,
        }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = executor.map(process_one, range(batch_size))
        for future in tqdm(futures, total=batch_size, desc="Pixel metrics (threaded)"):
            for key, value in future.items():
                result[key].append(value)
    # 2. 计算平均值
    for key in result.keys():
        result[key] = np.mean(result[key])
    return result


def load_image(image_path: Path):
    """
    读取图片文件, 并转为numpy
    :param image_path: 图片路径
    :return: numpy数组
    """
    # 如果image_path以~为开头, 自动解析
    if str(image_path).startswith("~"):
        image_path = os.path.expanduser(str(image_path))
    # 转为绝对路径
    image_path = Path(image_path).resolve()
    # 转为灰度图再读取
    assert image_path.exists(), f"Image path {image_path} does not exist."
    image = np.array(Image.open(image_path).convert("L"))
    image = image < 128  # 以128为阈值二值化
    return image


if __name__ == "__main__":
    # 测试代码
    start_time = time.time()
    gt = [load_image("/home/yuyangxin/data/dataset/MagicBrush/fake_mask/81964_mask.png")]
    pred = [load_image("/data0/yuyangxin/FakeShield/playground/MFLM_output/MagicBrush/81964-output1.png")]
    print(compute_pixel_metrics(gt, pred, use_sigmoid=False))
    print(f"耗时: {time.time() - start_time:.4f} 秒")
