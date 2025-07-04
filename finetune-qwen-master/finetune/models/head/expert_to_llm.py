import torch
import torch.nn as nn


class ExpertToLLM(nn.Module):
    """
    新版模型：
        - 先通过 AdaptiveAvgPool2d 将输入图像（例如 512×512）缩减到指定尺寸（默认为 16×16）
        - 随后展平并使用两层全连接层生成所需的 tokens
    输入: (batch_size, in_channels, image_size, image_size)
    输出: (batch_size, num_new_tokens, out_features)
    """

    def __init__(self, in_channels=64, num_new_tokens=32, out_features=2048):
        super().__init__()
        self.num_tokens = num_new_tokens
        self.out_dim = out_features

        # 使用自适应平均池化将输入大小固化为 pool_size x pool_size，降低全连接层的参数量
        input_dim = in_channels * num_new_tokens * num_new_tokens

        # 隐藏层维度，这里简单取输入维度的1/16，并确保至少为1
        hidden_dim = max(1, input_dim // 16)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((num_new_tokens, num_new_tokens)),
            nn.Flatten(),  # 展平后形状: (batch_size, in_channels * pool_size * pool_size)
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_new_tokens * out_features),
        )

    def forward(self, x):
        # x: (batch_size, in_channels, image_size, image_size)
        x = self.mlp(x)  # 输出形状: (batch_size, num_new_tokens * out_features)
        return x.view(-1, self.num_tokens, self.out_dim)  # 重塑为 (batch_size, num_new_tokens, out_features)


if __name__ == "__main__":
    # 测试代码
    model = ExpertToLLM()
    x = torch.randn(8, 3, 512, 512)  # 示例输入
    output = model(x)
    print(output.shape)  # 应输出 (8, 32, 2048)
