from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN


class MaskPolygonConverter:
    """
    高效的掩码与多边形转换工具类

    Attributes:
        max_points (int): 多边形顶点数上限（默认：1500）
        epsilon_ratio (float): 多边形简化精度比例（默认：0.001）
        min_points (int): 多边形顶点数下限（默认：3）
        unique (bool): 是否去重相同多边形（默认：True）
    """

    def __init__(
        self,
        max_points: int = 500,
        epsilon_ratio: float = 0.01,
        min_points: int = 3,
        unique: bool = True,
    ):
        self.max_points = max_points
        self.epsilon_ratio = epsilon_ratio
        self.min_points = min_points
        self.unique = unique

    def convert_to_polygons(
        self,
        mask: Union[Image.Image, np.ndarray, torch.Tensor],
        max_points: Optional[int] = None,
        epsilon_ratio: Optional[float] = None,
        min_points: Optional[int] = None,
        unique: Optional[bool] = None,
    ) -> List[List[int]]:
        """
        将掩码转换为优化的多边形表示

        Args:
            mask: 输入掩码（PIL.Image/np.ndarray）
            max_points: 顶点数上限（覆盖实例默认值）
            epsilon_ratio: 简化精度比例（覆盖实例默认值）
            min_points: 顶点数下限（覆盖实例默认值）
            unique: 是否去重（覆盖实例默认值）

        Returns:
            扁平化的多边形坐标列表 [[x1,y1,x2,y2,...], ...]
        """
        # 处理参数覆盖逻辑
        max_points = max_points or self.max_points
        epsilon_ratio = epsilon_ratio or self.epsilon_ratio
        min_points = min_points or self.min_points
        unique = unique if unique is not None else self.unique

        # 执行转换流程
        processed = self._preprocess_mask(mask)
        binarized = self._bin_mask(processed)
        contours = self._extract_contours(binarized)
        polygons = self._simplify_contours(contours, epsilon_ratio, min_points)

        controlled = self._control_points(polygons, max_points, min_points)
        return self._deduplicate(controlled) if unique else controlled

    def convert_to_bboxes(
        self,
        mask: Union[Image.Image, np.ndarray, torch.Tensor],
        max_boxes: int = 1,  # 默认值为1, 通过不同的呢容控制输入
        binarize: bool = True,  # 新增：是否先做二值化
    ) -> List[List[int]]:
        """
        将掩码转换为边界框表示，使用 k-means 将目标区域分割成多个小方框，
        尽可能规避 mask 中值为 0 的部分

        Args:
            mask: 输入掩码（PIL.Image/np.ndarray/torch.Tensor）
            max_boxes: 最大的边界框数量（默认：10）

        Returns:
            边界框坐标列表，每个边界框格式为 [x1, y1, x2, y2]
        """
        processed = self._preprocess_mask(mask)
        # 根据 binarize 参数决定是否进行二值化处理
        if binarize:
            # 这里是二值图
            target_map = self._bin_mask(processed)
        else:
            # 这里是灰度图"L"
            target_map = processed

        # 如果整张图像不存在非0像素，直接返回整张图像
        if target_map.min() != 0:
            h, w = target_map.shape
            return [[0, 0, w, h]]

        # 获取所有非0像素的位置
        pts = cv2.findNonZero(target_map)
        if pts is None:
            return []
        pts = pts.reshape(-1, 2).astype(np.float32)

        # 当目标点数较少或仅需要一个时，直接返回整体边界框
        if len(pts) == 0 or max_boxes < 2:
            x, y, w, h = cv2.boundingRect(pts)
            return [[x, y, x + w, y + h]]

        # 如果非零像素数量少于 max_boxes，则每个像素位置都构成一个小框（这里合并到同一框中）
        k = min(max_boxes, len(pts))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pts, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        bboxes = []
        for i in range(k):
            cluster = pts[labels.ravel() == i]
            if len(cluster) == 0:
                continue
            xi, yi, wi, hi = cv2.boundingRect(cluster.astype(np.int32))
            bboxes.append([xi, yi, xi + wi, yi + hi])

        # 最终统一 swap (col,row)->(row,col)
        return [[y1, x1, y2, x2] for x1, y1, x2, y2 in bboxes]

    def convert_to_bboxes_multi_objective(self, mask, max_boxes=5):
        processed = self._preprocess_mask(mask)
        binarized = self._bin_mask(processed)

        # 获取所有非零像素
        pts = cv2.findNonZero(binarized)
        if pts is None:
            return []
        pts = pts.reshape(-1, 2)

        # 基本情况处理
        if len(pts) == 0:
            return []
        if max_boxes == 1:
            x, y, w, h = cv2.boundingRect(pts)
            return [[x, y, x + w, y + h]]

        best_boxes = []
        best_iou = 0

        # 尝试不同的聚类数量找最优组合
        for k in range(1, max_boxes + 1):
            # K-means++聚类
            pts_float = pts.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
            _, labels, centers = cv2.kmeans(pts_float, k, None, criteria, 20, cv2.KMEANS_PP_CENTERS)

            # 计算每个聚类的框
            boxes = []
            for i in range(k):
                cluster_pts = pts[labels.flatten() == i]
                if len(cluster_pts) > 0:
                    x, y, w, h = cv2.boundingRect(cluster_pts)
                    boxes.append([x, y, x + w, y + h])

            # 生成重构掩码
            reconstructed = np.zeros_like(binarized)
            for box in boxes:
                x1, y1, x2, y2 = box
                reconstructed[y1:y2, x1:x2] = 255

            # 计算IoU
            intersection = np.logical_and(binarized > 0, reconstructed > 0).sum()
            union = np.logical_or(binarized > 0, reconstructed > 0).sum()
            iou = intersection / union if union > 0 else 0

            # 更新最优结果
            if iou > best_iou:
                best_iou = iou
                best_boxes = boxes

        return best_boxes

    def convert_to_bboxes_density(self, mask, max_boxes=5):
        processed = self._preprocess_mask(mask)
        binarized = self._bin_mask(processed)

        # 获取非零点
        pts = cv2.findNonZero(binarized)
        if pts is None:
            return []
        pts = pts.reshape(-1, 2).astype(np.float32)

        # 计算整体边界框
        if len(pts) == 0 or max_boxes < 2:
            x, y, w, h = cv2.boundingRect(pts)
            return [[x, y, x + w, y + h]]

        # 自适应设置参数
        pts_density = len(pts) / (binarized.shape[0] * binarized.shape[1])
        eps = max(5, min(30, int(20 / pts_density**0.5)))  # 自适应邻域大小

        # 执行聚类
        db = DBSCAN(eps=eps, min_samples=5).fit(pts)
        labels = db.labels_

        # 处理聚类结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters = min(n_clusters, max_boxes)

        if n_clusters == 0:
            x, y, w, h = cv2.boundingRect(pts)
            return [[x, y, x + w, y + h]]

        # 为每个簇计算边界框
        bboxes = []
        for i in range(n_clusters):
            cluster_pts = pts[labels == i]
            if len(cluster_pts) > 0:
                x, y, w, h = cv2.boundingRect(cluster_pts.astype(np.int32))
                bboxes.append([x, y, x + w, y + h])

        # 如果簇数量不足，尝试分割最大的边界框
        while len(bboxes) < max_boxes and bboxes:
            # 找到最大的边界框
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bboxes]
            largest_idx = areas.index(max(areas))
            x1, y1, x2, y2 = bboxes[largest_idx]

            # 水平或垂直分割，取决于长宽比
            w, h = x2 - x1, y2 - y1
            if w > h:  # 水平分割
                bboxes[largest_idx] = [x1, y1, x1 + w // 2, y2]
                bboxes.append([x1 + w // 2, y1, x2, y2])
            else:  # 垂直分割
                bboxes[largest_idx] = [x1, y1, x2, y1 + h // 2]
                bboxes.append([x1, y1 + h // 2, x2, y2])

        return bboxes

    def convert_bboxes_to_mask(self, bboxes: List[List[int]], image_size: tuple) -> Image.Image:
        """
        将边界框列表转换回二值掩码图像

        Args:
            bboxes: 边界框坐标列表，每个边界框格式为 [x1, y1, x2, y2]
            image_size: 输出图像尺寸 (width, height)

        Returns:
            PIL.Image: 二值掩码图像（模式L，0-255）
        """
        canvas = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=255, thickness=-1)
        return Image.fromarray(canvas, mode="L")

    def convert_to_mask(self, polygons: List[List[int]], image_size: tuple) -> Image.Image:
        """
        将多边形转换回掩码图像

        Args:
            polygons: 多边形坐标列表 [[x1,y1,x2,y2,...], ...]
            image_size: 输出图像尺寸 (width, height)

        Returns:
            PIL.Image: 二值掩码图（模式L，0-255）
        """
        canvas = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(canvas, [pts], color=255)
        return Image.fromarray(canvas, mode="L")

    def _preprocess_mask(self, mask: Union[Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
        """统一转换为uint8格式的灰度数组"""
        if isinstance(mask, Image.Image):
            if mask.mode == "1":
                return np.array(mask).astype(np.uint8) * 255
            else:
                return np.array(mask.convert("L")).astype(np.uint8)
        elif isinstance(mask, torch.Tensor):
            mask = mask.numpy()
            return mask.astype(np.uint8) * 255 if mask.max() <= 1 else mask.astype(np.uint8)
        elif isinstance(mask, np.ndarray):
            if mask.dtype != np.uint8:
                return (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
            else:
                return mask
        else:
            raise TypeError("Unsupported mask type: {}".format(type(mask)))

    def _bin_mask(self, mask: np.ndarray) -> np.ndarray:
        """二值化处理（自动处理不同输入范围）"""
        if mask.max() <= 1:
            return (mask * 255).astype(np.uint8)
        return cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    def _extract_contours(self, mask: np.ndarray) -> list:
        """提取并排序轮廓（按面积降序）"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=lambda c: (-cv2.contourArea(c), *cv2.boundingRect(c)[1::-1]))

    def _simplify_contours(self, contours: list, epsilon_ratio: float, min_points: int) -> list:
        """轮廓简化主逻辑"""
        simplified = []
        for contour in contours:
            epsilon = epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= min_points:
                # 将多边形坐标扁平化
                simplified.append(approx.reshape(-1, 2).astype(int).ravel().tolist())
        return simplified

    def _control_points(self, polygons: list, max_points: int, min_points: int) -> list:
        """总顶点数控制策略"""
        total_points = sum(len(p) // 2 for p in polygons)
        if total_points <= max_points:
            return polygons

        ratio = max_points / total_points
        optimized = []
        for poly in polygons:
            pts = np.array(poly).reshape(-1, 2)
            target = max(int(len(pts) * ratio), min_points)
            optimized.append(self._optimize_polygon(pts, target, min_points))
        return optimized

    def _optimize_polygon(self, points: np.ndarray, target: int, min_points: int) -> list:
        """基于二分法的多边形顶点优化"""
        contour = points.astype(np.float32).reshape(-1, 1, 2)
        if len(points) <= target:
            return points.ravel().tolist()

        epsilon_range = (0.0, cv2.arcLength(contour, True))
        best_epsilon = 0.0
        for _ in range(50):  # 有限次数的二分查找
            epsilon = sum(epsilon_range) / 2
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) <= target:
                epsilon_range = (epsilon_range[0], epsilon)
                best_epsilon = epsilon
            else:
                epsilon_range = (epsilon, epsilon_range[1])

        # 最终优化验证
        optimized = cv2.approxPolyDP(contour, best_epsilon, True)
        while len(optimized) > target and len(optimized) > min_points:
            best_epsilon += 1.0
            optimized = cv2.approxPolyDP(contour, best_epsilon, True)
        return optimized.reshape(-1, 2).astype(int).ravel().tolist()

    def _deduplicate(self, polygons: list) -> list:
        """顺序保持型去重"""
        seen = set()
        return [p for p in polygons if not (tuple(p) in seen or seen.add(tuple(p)))]


if __name__ == "__main__":
    converter = MaskPolygonConverter(max_points=1000)
    mask = np.zeros((5, 5), dtype=np.uint8)
    pts = [(1, 1), (3, 4)]
    for x, y in pts:
        mask[x, y] = 255
    print("Original Mask:\n", mask)

    res = converter.convert_to_bboxes(mask, max_boxes=2)
    # res标注[[1, 1, 2, 2], [4, 3, 5, 4]]
    # 期望标注 {(1, 1, 2, 2), (3, 4, 4, 5)}
    expected = {(1, 1, 2, 2), (3, 4, 4, 5)}
