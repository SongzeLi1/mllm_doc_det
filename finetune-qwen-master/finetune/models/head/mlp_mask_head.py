import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPMaskHead(nn.Module):
    def __init__(self, num_token=32, hidden_dim=2048, output_size=1024, bottleneck_dim=512, rank=64):
        """
        输入: [batch_size, num_token, hidden_dim]
        输出: [batch_size, output_size, output_size]
        采用低秩分解来减少全连接层的参数量，即先映射到较小的 bottleneck 表示，
        再分解学习两个低秩矩阵 A 和 B，使得 mask = A @ B
        """
        super().__init__()
        self.output_size = output_size
        self.rank = rank
        # 先将所有 token 拼接后映射到 bottleneck 空间
        self.fc_reduce = nn.Sequential(
            nn.Linear(num_token * hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        # 分别将 bottleneck 表示映射为 A 和 B
        self.fc_a = nn.Linear(bottleneck_dim, output_size * rank)
        self.fc_b = nn.Linear(bottleneck_dim, rank * output_size)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
        self.ds_grads_remaining = 0  # 为 DeepSpeed Zero3 添加必要属性

    def _init_weights(self):
        # 对线性层进行权重初始化
        nn.init.kaiming_uniform_(self.fc_reduce[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.fc_reduce[0].bias)
        nn.init.xavier_uniform_(self.fc_a.weight)
        nn.init.zeros_(self.fc_a.bias)
        nn.init.xavier_uniform_(self.fc_b.weight)
        nn.init.zeros_(self.fc_b.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, num_token, hidden_dim]
        B, num_token, hidden_dim = x.shape
        x = x.flatten(1)  # 将除 batch 外的维度展平，确保内存连续性
        z = self.fc_reduce(x)
        a = self.fc_a(z).view(B, self.output_size, self.rank)
        b = self.fc_b(z).view(B, self.rank, self.output_size)
        mask = torch.bmm(a, b)  # [B, output_size, output_size]
        return mask


if __name__ == "__main__":
    model = MLPMaskHead()
    x = torch.randn(2, 32, 2048)
    output = model(x)
    print(output.shape)
