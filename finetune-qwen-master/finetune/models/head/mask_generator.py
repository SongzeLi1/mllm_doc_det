import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGenerator(nn.Module):
    def __init__(self, num_token=32, hidden_dim=2048, output_size=1024, num_heads=8, expansion_ratio=4, dropout=0.1):
        super().__init__()
        self.num_token = num_token
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # 位置编码增强
        self.pos_embed = nn.Parameter(torch.randn(1, num_token, hidden_dim))

        # 多头注意力模块
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 空间解码器
        self.decoder = nn.Sequential(
            # 特征展开
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # 多阶段上采样
            UpBlock(hidden_dim // 2, 512, scale_factor=2),
            UpBlock(512, 256, scale_factor=2),
            UpBlock(256, 128, scale_factor=2),
            UpBlock(128, 64, scale_factor=2),
            # 最终投影
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # 动态尺寸适配
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)
        self._init_weights()

    def _init_weights(self):
        # 参数初始化策略[1,2](@ref)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状: (B, num_token, hidden_dim)
        B = x.size(0)

        # 增强位置编码[3](@ref)
        x = x + self.pos_embed

        # 自注意力处理
        attn_output, _ = self.self_attn(x, x, x)  # (B, 32, 2048)

        # 空间特征重塑
        spatial_feat = attn_output.transpose(1, 2).view(B, -1, 4, 8)  # (B, 2048, 4, 8)

        # 多尺度上采样
        mask = self.decoder(spatial_feat)

        # 动态尺寸调整
        if mask.shape[-2:] != (self.output_size, self.output_size):
            mask = F.interpolate(mask, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)

        return mask.squeeze(1)


class UpBlock(nn.Module):
    """上采样模块"""

    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
