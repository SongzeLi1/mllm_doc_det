import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, use_sigmoid=False, smooth=1e-6, loss_weight=1.0, reduction="mean"):
        """
        Dice损失函数，仅支持二分类
        Args:
            use_sigmoid: 是否使用sigmoid激活（适用于二分类）
            smooth: 平滑项，防止分母为零
            loss_weight: 损失权重系数
            reduction: 批次结果的归约方式，可选'mean'或'none'
        """
        super().__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # 输入检查
        if not torch.is_tensor(y_pred) or not torch.is_tensor(y_true):
            raise TypeError("输入必须是PyTorch张量")

        # 处理激活函数
        if self.use_sigmoid:
            y_pred = torch.sigmoid(y_pred)

        # 针对不同维度的处理
        if y_pred.dim() == 3:  # 形状为(B, H, W)的二分类情况
            # 计算交集和并集
            intersection = torch.sum(y_pred * y_true, dim=(1, 2))
            union = torch.sum(y_pred, dim=(1, 2)) + torch.sum(y_true, dim=(1, 2))
            # 计算Dice系数
            dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            raise ValueError(f"不支持的输入维度: {y_pred.dim()}")

        # 计算损失
        dice_loss = 1 - dice_coeff

        # 应用归约方法
        if self.reduction == "mean":
            dice_loss = dice_loss.mean()

        return dice_loss * self.loss_weight
