import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClsHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512], output_dim=1, dropout_rate=0.5):
        """
        MLP 分类头模型
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dropout_rate: dropout概率
        """
        super(MLPClsHead, self).__init__()
        layers = []
        prev_dim = input_dim

        # 构建MLP网络
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, output_dim)
        # self.alpha = nn.Parameter(torch.tensor(2.0))  # 定义可训练参数 self.a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """处理输入维度 [batch, 2, input_dim] -> [batch, output_dim]"""
        x = x.mean(dim=1)  # 对第二维进行平均，合并成 [batch, input_dim]
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = x.squeeze(-1)  # 压缩输出中的单一维度
        return x
