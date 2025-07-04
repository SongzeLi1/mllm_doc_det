import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyWithLabelSmoothing(nn.Module):
    def __init__(
        self,
        pos_weight=0.5,
        neg_weight=1.0,
        alpha=0.1,
        loss_weight=1.0,
        use_sigmoid=False,
        use_label_smoothing=False,
    ):
        """
        带标签平滑和正样本加权的二分类交叉熵损失
        :param pos_weight: 正样本的权重
        :param alpha: 标签平滑参数，仅在 use_label_smoothing 为 True 时生效
        :param loss_weight: 整体损失的权重
        :param use_label_smoothing: 是否应用标签平滑开关
        """
        super().__init__()
        if use_label_smoothing:
            assert 0 <= alpha <= 1.0, "alpha必须在[0, 1]范围内"
        assert pos_weight > 0, "pos_weight需为正数"
        assert neg_weight > 0, "neg_weight需为正数"

        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

        self.alpha = alpha
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.use_label_smoothing = use_label_smoothing

    def forward(self, pred_label, gt_label):
        # 根据开关决定是否应用标签平滑
        if self.use_label_smoothing:
            smoothed_targets = gt_label * (1 - self.alpha) + (1 - gt_label) * self.alpha
        else:
            smoothed_targets = gt_label.float()

        # 创建样本权重, 正样本权重为 pos_weight, 负样本权重为 neg_weight
        # 正样本标签为0, 负样本标签为1
        weights = torch.where(gt_label == 0, self.pos_weight, self.neg_weight).float()

        # 计算带权重的BCE损失
        if self.use_sigmoid:
            loss = F.binary_cross_entropy_with_logits(pred_label, smoothed_targets, weight=weights)
        else:
            loss = F.binary_cross_entropy(pred_label, smoothed_targets, weight=weights)

        # 应用整体损失权重
        return loss * self.loss_weight
