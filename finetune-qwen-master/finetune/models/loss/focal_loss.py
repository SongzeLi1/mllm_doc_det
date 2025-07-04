import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0):
        """Focal Loss实现
        假设α设置为较高的值，比如接近1，那么正类的损失权重会增加，这可能有助于缓解正类样本少的问题。
        例如，如果真实样本（正类）很少，设置α=0.75可能会使模型更关注正类样本。
        ​α值越大：正类（少数类，如真实样本）的损失权重更高，模型更关注正类样本的分类效果。
        ​α值越小：负类（多数类，如伪造样本）的损失权重更高，模型偏向优化负类。
        gamma越大: 易分样本（预测概率接近0或1的样本）的损失贡献显著降低。
        gamma越小: 易分样本的损失贡献相对较高，模型对所有样本一视同仁（接近标准交叉熵）。
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        # 自动设备兼容
        if pred.device != target.device:
            target = target.to(pred.device)
        # 计算焦点损失
        # 确保输入维度兼容
        if pred.dim() != target.dim():
            target = target.unsqueeze(1)

        # 计算sigmoid预测
        pred_sigmoid = torch.sigmoid(pred)
        target = target.type_as(pred)

        # 计算平衡因子和调制因子
        pt = torch.where(target == 1, 1 - pred_sigmoid, pred_sigmoid)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)

        # 计算bce损失
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # 应用焦点权重
        loss = bce_loss * focal_weight

        # 归约策略
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss
