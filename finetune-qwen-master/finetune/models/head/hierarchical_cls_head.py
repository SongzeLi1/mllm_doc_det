import torch
import torch.nn as nn


class HierarchicalCLSHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.token_proj = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU())
        self.transformer = nn.TransformerEncoderLayer(256, 4, 1024)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 2)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: [B,32,hidden_size]
        token_feat = self.token_proj(x)  # [B,32,256]
        context_feat = self.transformer(token_feat)  # 上下文交互
        pooled = self.pool(context_feat.transpose(1, 2))  # [B,256,1]
        return self.fc(pooled.squeeze(-1))  # [B,2]
