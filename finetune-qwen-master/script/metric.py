"""
评估初始大模型的基本指标
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from finetune.plugin.metrics import (
    compute_label_metrics,
    compute_pixel_metrics,
    load_image,
)
from finetune.utils.json_decoder import JsonDecoder


def get_pred_label(value: dict):
    # key全部转为小写
    value = {k.lower(): v for k, v in value.items()}
    # 检查"result"键
    pred_label = value.get("result")
    if pred_label is not None:
        if pred_label == "fake" or pred_label == "faked" or pred_label == "true":
            return 1
        elif pred_label == "real" or pred_label == "true":
            return 0
        else:
            return -1

    # 检查"is_fake"键
    pred_label = value.get("is_fake")
    if pred_label is not None:
        return 1 if pred_label is True else 0

    # 检查"is_real"键
    pred_label = value.get("is_real")
    if pred_label is not None:
        return 0 if pred_label is True else 1

    # 检查"image"键
    image = value.get("image")
    if image is not None:
        if isinstance(image, dict):
            if image.get("is_fake") is not None:
                return 1 if image.get("is_fake") is True else 0
            if image.get("is_real") is not None:
                return 1 if image.get("is_real") is False else 0
        if isinstance(image, str):
            if image == "fake" or image == "false" or image == "faked":
                return 1
            elif image == "real":
                return 0
        if isinstance(image, bool):
            return 1 if image is False else 0

    # 检查"output"键
    output = value.get("output")
    if output is not None:
        if isinstance(output, str):
            if output == "fake" or output == "false" or output == "faked" or output == "faded":
                return 1
            elif output == "real":
                return 0
        if isinstance(output, list):
            if len(output) == 0:
                return None
            elif isinstance(output[0], str):
                if output[0] == "fake" or output[0] == "false" or output[0] == "faked":
                    return 1
                elif output[0] == "real":
                    return 0
            elif output[0].get("label") is not None:
                return 1 if output[0].get("is_fake") is True else 0
            elif output[0].get("result") == "fake":
                return 1
            elif output[0].get("result") == "real":
                return 0
        if isinstance(output, bool):
            return 1 if output is False else 0

    # 检查conclusion键
    conclusion = value.get("conclusion")
    if conclusion is not None and isinstance(conclusion, dict):
        if conclusion.get("is_fake") is not None:
            return 1 if conclusion.get("is_fake") is True else 0
        if conclusion.get("is_real") is not None:
            return 0 if conclusion.get("is_real") is True else 1

    content = value.get("content", "")
    if any(
        term in content
        for term in [
            '"result": "fake',
            '"result": "false',
            '"result": "faded',
            "fake",
            "have been manipulated or edited",
        ]
    ):
        return 1
    elif any(term in content for term in ['"result": "real"', '"result": "true"', "'result': 'real'"]):
        return 0

    # 如果都没有找到相关键
    return None


def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_np(i) for i in obj]
    return obj


def save_for_json(data, save_path):
    save_path = Path(save_path)
    target = {}
    if save_path.exists():
        # 判断文件是有内容的
        if os.path.getsize(save_path) > 0:
            target = JsonDecoder.load_json_file(save_path)
    target.update(data)
    with open(save_path, "w") as f:
        json.dump(convert_np(target), f, indent=4)


def get_label_cls(data):
    gt_labels, pred_labels = [], []
    for key, value in data.items():
        pred_labels.append(value["pred_label"])
        gt_labels.append(value["gt_label"])
    return gt_labels, pred_labels


def get_label(data):
    gt_labels, pred_labels = [], []
    for key, value in data.items():
        pred_label = get_pred_label(value)
        if pred_label is None:
            if value.get("pred_label") is not None:
                pred_label = 1 if value["pred_label"] > 0.5 else 0
            else:
                raise ValueError(f"未找到有效的预测标签: {key}")
        pred_labels.append(pred_label)
        gt_labels.append(value["gt_label"])
    return gt_labels, pred_labels


def metric_mask(datas, parent_dir="."):
    ret = {}
    for json_name, data in datas.items():
        gt_mask_batch, pred_mask_batch = [], []
        # 使用 tqdm 显示处理进度
        for file_name, value in tqdm(data.items(), desc=f"{json_name} 处理图片进度", total=len(data), unit="file"):
            if value["pred_mask_path"] is None or value["gt_mask_path"] is None:
                continue
            pred_mask_path = Path(value["pred_mask_path"])
            if not pred_mask_path.is_absolute():
                pred_mask_path = Path(parent_dir) / pred_mask_path
            try:
                pred_mask = load_image(pred_mask_path)
            except Exception as e:
                print(f"加载预测mask失败: {pred_mask_path}, 错误: {e}")
                continue
            if value["gt_mask_path"] == "positive":
                # 创造一个和pred_mask一样大小的全0 mask
                gt_mask = np.zeros_like(pred_mask)
            else:
                # 正常读取 gt_mask
                gt_mask_path = Path(value["gt_mask_path"])
                if not gt_mask_path.is_absolute():
                    gt_mask_path = Path(parent_dir) / gt_mask_path
                gt_mask = load_image(gt_mask_path)
            gt_mask_batch.append(gt_mask)
            pred_mask_batch.append(pred_mask)
        if len(gt_mask_batch) == 0:
            print(f"[{json_name}]没有找到有效的mask数据")
            ret[json_name] = {
                "seg": "没有找到有效的gt mask数据",
            }
        # 结果转为4位小数
        print(f"开始统计的像素级指标: {json_name}")
        if len(gt_mask_batch) == 0:
            ret[json_name] = {
                "seg": "没有找到有效的gt mask数据",
            }
            continue
        res = compute_pixel_metrics(gt_mask_batch, pred_mask_batch, use_sigmoid=False)
        ret[json_name] = res
    return ret


def metric_cls(datas, get_label_func=get_label_cls):
    res = {}
    for json_file, data in datas.items():
        gt_labels, pred_labels = get_label_func(data)
        if None in pred_labels:
            res[json_file] = {"cls": "未发现二分类检测结果"}
        else:
            ret = compute_label_metrics(np.array(gt_labels), np.array(pred_labels))
            res[json_file] = ret
    return res


def metric(json_path, save_path, get_label_func=get_label, parent_dir="."):
    json_path = Path(json_path)
    # 判断json_path是文件还是目录
    datas = {}
    if json_path.is_dir():
        # 遍历目录下所有json文件和jsonl文件
        for file in json_path.glob("*.json*"):
            if file.name == "summary.json":
                continue
            datas[file.name] = JsonDecoder.load_file(file)
    else:
        datas[json_path.name] = JsonDecoder.load_file(json_path)

    res = {}
    cls_res = metric_cls(datas, get_label_func)
    pred_res = metric_mask(datas, parent_dir)
    # 合并结果
    for key, value in cls_res.items():
        if key not in res:
            res[key] = {}
        res[key].update(value)
    for key, value in pred_res.items():
        if key not in res:
            res[key] = {}
        res[key].update(value)

    save_path = Path(save_path)
    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    save_for_json(res, save_path)
    print(f"分析结果已保存到: {save_path}")
    print(f"分析结果: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估初始大模型的基本指标")
    parser.add_argument("--parent_dir", type=str, default=".", help="父目录路径")
    parser.add_argument("--analysis_path", type=str, required=True, help="待分析的json或目录路径")
    parser.add_argument("--save_path", type=str, required=True, help="分析结果保存路径")
    parser.add_argument("--cls_label", action="store_true", help="是否使用cls标签")
    args = parser.parse_args()

    start_time = time.time()
    if args.cls_label:
        metric(args.analysis_path, args.save_path, get_label_func=get_label_cls, parent_dir=args.parent_dir)
    else:
        metric(args.analysis_path, args.save_path, get_label_func=get_label, parent_dir=args.parent_dir)
    end_time = time.time()
    print(f"分析耗时: {end_time - start_time:.4f}秒")
