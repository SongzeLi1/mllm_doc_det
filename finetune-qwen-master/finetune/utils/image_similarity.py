import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
import torch
import lpips


class ImageSimilarity:
    LPIPS_MODEL = lpips.LPIPS(net="vgg", spatial=True)



    @staticmethod
    def load_image_array(img: Union[Image.Image, str, Path, np.ndarray], color_space: str = "RGB") -> np.ndarray:
        # 支持 numpy array
        if isinstance(img, np.ndarray):
            arr = img
        # 支持 PIL Image
        elif isinstance(img, Image.Image):
            arr = np.array(img)
        # 支持 网络 URL / Base64 / 本地路径
        elif isinstance(img, str) or isinstance(img, Path):
            img = str(img)  # 确保是字符串类型
            parsed = urlparse(img)
            # 网络 URL
            if parsed.scheme in ("http", "https"):
                resp = requests.get(img)
                if resp.status_code != 200:
                    raise ValueError(f"无法加载网络图像: {img}")
                img_pil = Image.open(BytesIO(resp.content)).convert("RGB")
                arr = np.array(img_pil)
            # Base64 Data URI
            elif img.strip().startswith("data:"):
                header, b64 = img.split(",", 1)
                img_pil = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                arr = np.array(img_pil)
            # 本地文件路径
            else:
                arr = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise ValueError(f"无法加载图像: {img}")
        else:
            raise ValueError("未知的图像类型。支持 PIL.Image.Image、str(URL/Base64/Path)、Path、np.ndarray。")

        # 如果是 PIL 转换或 cv2 读取后的单通道或四通道，先转为 BGR 3通道
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        color_space = color_space.upper()
        if color_space == "HSV":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        elif color_space == "RGB":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif color_space in ("GRAY", "L"):
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            # 保持三维 `(H, W, 1)`
            arr = gray[:, :, np.newaxis]
        return arr

    @staticmethod
    def compare_images_mse(
        src: Union[Image.Image, str, Path, np.ndarray],
        tgt: Union[Image.Image, str, Path, np.ndarray],
        ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]] = None,
        color_space: str = "RGB",
        norm: str = "zscore",
        heatmap: bool = False,
    ) -> Union[np.ndarray, Image.Image]:
        """基于逐像素 MSE 的差异图计算。"""
        sa = ImageSimilarity.load_image_array(src, color_space).astype(np.float32)
        ta = ImageSimilarity.load_image_array(tgt, color_space).astype(np.float32)
        if sa.shape != ta.shape:
            raise ValueError("源图像和目标图像尺寸不一致。")

        # 计算逐像素 MSE，并归一化到 [0,1]
        mse_map = np.mean((sa - ta) ** 2, axis=-1) / (255.0**2)
        diff_gray = ImageSimilarity.norm_method(mse_map, norm)
        diff_gray = ImageSimilarity.apply_ref_mask(diff_gray, ref_mask)
        if heatmap:
            return ImageSimilarity.to_heatmap(diff_gray)
        return diff_gray

    @staticmethod
    def compare_images_pixel(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]] = None,
        color_space: str = "RGB",
        norm: str = "zscore",
        heatmap: bool = False,
        gray: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        src_array = ImageSimilarity.load_image_array(src_img, color_space)
        target_array = ImageSimilarity.load_image_array(target_img, color_space)
        if src_array.shape != target_array.shape:
            raise ValueError("源图像和目标图像的尺寸不匹配。")

        if len(src_array.shape) == 3:
            if color_space.upper() == "HSV":
                # 一次转换到 int16 防止溢出
                sa = src_array.astype(np.int16)
                ta = target_array.astype(np.int16)
                # Hue 通道差值，考虑环绕（0–180），并归一化到 [0,1]
                dh = np.abs(sa[:, :, 0] - ta[:, :, 0])
                h_norm = (np.minimum(dh, 180 - dh).astype(np.float32)) / 180.0
                # S、V 通道差值并归一化到 [0,1]
                sv_diff = cv2.absdiff(src_array[:, :, 1:], target_array[:, :, 1:]).astype(np.float32) / 255.0
                # 对 H、S、V 三路差异取平均
                diff_gray = (h_norm + sv_diff[:, :, 0] + sv_diff[:, :, 1]) / 3.0
                diff_gray = diff_gray * 255
            elif color_space.upper() == "RGB":
                diff = cv2.absdiff(src_array, target_array).astype(np.float32)
                diff_gray = diff.mean(axis=2)
        elif color_space.upper() == "GRAY":
            # 保持 float32 精度
            diff_gray = cv2.absdiff(src_array, target_array).astype(np.float32)
        else:
            raise ValueError(f"不支持的颜色空间差异计算: {color_space}")

        # 归一化
        diff_gray = ImageSimilarity.norm_method(diff_gray, norm)
        diff_gray = ImageSimilarity.apply_ref_mask(diff_gray, ref_mask)
        if heatmap:
            return ImageSimilarity.to_heatmap(diff_gray)
        elif gray:
            return ImageSimilarity.to_grey(diff_gray)
        return diff_gray

    @staticmethod
    def compare_images_ssim(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]] = None,
        norm: str = "zscore",
        heatmap: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        src_array = ImageSimilarity.load_image_array(src_img)
        target_array = ImageSimilarity.load_image_array(target_img)
        if src_array.shape != target_array.shape:
            raise ValueError("源图像和目标图像的尺寸不一致。")

        # 转换到灰度
        src_gray = cv2.cvtColor(src_array, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)
        _, diff_map = ssim(src_gray, target_gray, full=True)
        # 将相似度映射转为差异（0—1）
        diff_map = 1.0 - diff_map
        # 归一化
        diff_gray = ImageSimilarity.norm_method(diff_map, norm)
        diff_gray = ImageSimilarity.apply_ref_mask(diff_gray, ref_mask)
        if heatmap:
            return ImageSimilarity.to_heatmap(diff_gray)
        return diff_gray

    @staticmethod
    def check_and_resize(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
    ) -> Image.Image:
        """
        检查 src_img 和 target_img 的尺寸是否一致。
        如果不一致，则将 target_img 调整到与 src_img 相同的尺寸并返回调整后的数组；
        否则返回 None。
        """
        src_array = ImageSimilarity.load_image_array(src_img)
        tgt_array = ImageSimilarity.load_image_array(target_img)
        if src_array.shape != tgt_array.shape:
            h, w = src_array.shape[:2]
            tgt_array = cv2.resize(tgt_array, (w, h), interpolation=cv2.INTER_AREA)
        return Image.fromarray(tgt_array.astype(np.uint8), mode="RGB")

    @staticmethod
    def compare_images_lpips(
        img_real: Union[Image.Image, str, Path, np.ndarray],
        img_fake: Union[Image.Image, str, Path, np.ndarray],
        ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]] = None,
        norm="zscore",
        heatmap: bool = False,
        gray: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        """
        使用 LPIPS (Learned Perceptual Image Patch Similarity) 库比较两张图像的相似度。
        LPIPS 通过​​深度特征空间的距离计算​​模拟人类视觉系统，使评估结果更贴近主观感知
        CODE: https://github.com/richzhang/PerceptualSimilarity
        """
        src_array = ImageSimilarity.load_image_array(img_real)
        target_array = ImageSimilarity.load_image_array(img_fake)
        if src_array.shape != target_array.shape:
            raise ValueError("源图像和目标图像的尺寸不一致。")

        src_tensor = lpips.im2tensor(src_array)
        target_tensor = lpips.im2tensor(target_array)

        with torch.no_grad():
            score_map_tensor = ImageSimilarity.LPIPS_MODEL.forward(src_tensor, target_tensor)

        # 归一化
        diff_gray = ImageSimilarity.norm_method(score_map_tensor, norm)
        diff_gray = ImageSimilarity.apply_ref_mask(diff_gray, ref_mask)
        if heatmap:
            return ImageSimilarity.to_heatmap(diff_gray)
        elif gray:
            return ImageSimilarity.to_grey(diff_gray)
        return diff_gray

    @staticmethod
    def diff_weight(
        src_np: Union[Image.Image, str, Path, np.ndarray],
        target_np: Union[Image.Image, str, Path, np.ndarray],
        heatmap: bool = False,
    ) -> Union[np.ndarray, Image.Image]:
        """
        计算两张矩阵之间的差异，添加权重, src的值为0-1之间的小数, 越接近1越权重weight越大
        """
        # 计算加权差异
        diff = np.abs(src_np - target_np)
        weighted = diff * src_np
        if heatmap:
            return ImageSimilarity.to_heatmap(weighted)
        # 转为 PIL 灰度图，模式 “L”
        gray_uint8 = (weighted * 255).astype(np.uint8)
        # 如果多通道，把最后一维 squeeze 掉
        gray_uint8 = gray_uint8.squeeze()
        return Image.fromarray(gray_uint8, mode="L")
        # return weighted

    @staticmethod
    def threshold_gray(src_np: np.ndarray, threshold_gray: float) -> Union[np.ndarray, Image.Image]:
        """
        将灰度图像应用阈值处理，返回二值化图像。
        :param src_np: 输入的灰度图像数组
        :param threshold_gray: 阈值，范围 [0, 1]
        :return: 二值化后的图像数组或 PIL 图像
        """
        # 根据阈值, 将src_np 转为 0, 1
        # src_np的值为0-1之间的浮点数
        target_np = np.where(src_np >= threshold_gray, 1.0, 0.0).astype(np.float32)
        # 返回灰度图
        gray_uint8 = (target_np * 255).astype(np.uint8)
        return Image.fromarray(gray_uint8, mode="L")

    @staticmethod
    def norm_method(score_map: Union[torch.Tensor, np.ndarray], norm_method: str = "sigmoid", eps=1e-8) -> np.ndarray:
        # 兼容 numpy array 输入
        if isinstance(score_map, np.ndarray):
            score_map = torch.from_numpy(score_map)
            score_map = score_map.to(torch.float32)
        if norm_method == "minmax":
            score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + eps)
        elif norm_method == "zscore":
            score_map = (score_map - score_map.mean()) / (score_map.std() + eps)
            score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + eps)
        elif norm_method == "percentile":
            lo, hi = torch.quantile(score_map, torch.tensor([0.05, 0.95], dtype=torch.float32))
            score_map = torch.clamp(score_map, min=lo, max=hi)
            score_map = (score_map - lo) / (hi - lo + eps)
        elif norm_method == "sigmoid":
            score_map = torch.sigmoid(score_map)
        elif norm_method is None:
            score_map = score_map
        else:
            raise ValueError(f"未知的归一化方法: {norm_method}")

        # 移除多余的维度
        score_map = score_map.squeeze()
        return score_map.cpu().numpy() if torch.is_tensor(score_map) else score_map

    @staticmethod
    def apply_ref_mask(diff: np.ndarray, ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]]) -> np.ndarray:
        if ref_mask is not None:
            m = ImageSimilarity.load_image_array(ref_mask, "GRAY")
            _, mb = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            diff[mb.astype(bool)] = diff.max()
        return diff

    @staticmethod
    def to_heatmap(diff: np.ndarray) -> Image.Image:
        # 断言 diff 是归一化到 [0, 1] 的数组
        if not (0 <= diff.min() <= 1 and 0 <= diff.max() <= 1):
            raise ValueError("差异图必须是归一化到 [0, 1] 的数组。")
        c_map = (diff * 255).astype(np.uint8)
        heat = cv2.applyColorMap(c_map, cv2.COLORMAP_JET)
        return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))

    @staticmethod
    def to_grey(diff: np.ndarray) -> Image.Image:
        # 断言 diff 是归一化到 [0, 1] 的数组
        if not (0 <= diff.min() <= 1 and 0 <= diff.max() <= 1):
            raise ValueError("差异图必须是归一化到 [0, 1] 的数组。")
        # 差异越大越接近白色, 差异越小越接近黑色
        # 直接把归一化后的值映射到 [0,255]，并生成 L 模式灰度图
        c_map = (diff * 255).round().astype(np.uint8)
        return Image.fromarray(c_map, mode="L")
