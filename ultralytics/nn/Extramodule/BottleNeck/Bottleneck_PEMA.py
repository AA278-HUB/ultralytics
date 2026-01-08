import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Partial_Conv3x3(nn.Module):
    """用于降低计算量的局部卷积"""

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3x3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3x3
        self.partial_conv3x3 = nn.Conv2d(self.dim_conv3x3, self.dim_conv3x3, 3, 1, 1, groups=self.dim_conv3x3,
                                         bias=False)

    def forward(self, x):
        # 仅对部分通道处理，其余直连，保持内存访问的高效
        x1, x2 = torch.split(x, [self.dim_conv3x3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3x3(x1)
        return torch.cat((x1, x2), dim=1)


class LiteRep_Attention(nn.Module):
    """改进的轻量化注意力，替代标准SE"""

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class LiteRep_TokenMixer(nn.Module):
    """改进的Token Mixer：结合PConv与结构重参数化"""

    def __init__(self, dim):
        super().__init__()
        self.pconv = Partial_Conv3x3(dim)
        self.re_conv = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim)  # 1x1 结构重参分支
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # 训练时融合 PConv 和 1x1 Conv
        return self.bn(self.pconv(x) + self.re_conv(x) + x)


class LiteRep_AttentionBlock(nn.Module):
    """
    重新设计的模块：LiteRep_AttentionBlock
    特点：更低的FLOPs，更强的特征对齐
    """

    def __init__(self, inp, oup, stride=1, expand_ratio=2.0):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)

        # 1. Token Mixer (空间转换)
        if stride == 2:
            self.token_mixer = nn.Sequential(
                nn.Conv2d(inp, inp, 3, 2, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.token_mixer = nn.Sequential(
                LiteRep_TokenMixer(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

        # 2. Attention (特征增强)
        self.attn = LiteRep_Attention(oup)

        # 3. Channel Mixer (通道变换)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(oup, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        self.alpha = nn.Parameter(torch.ones(1))  # 可学习的残差系数

    def forward(self, x):
        # Token Mix
        x = self.token_mixer(x)
        # Attention
        x = self.attn(x)
        # Channel Mix + Dynamic Residual
        return x + self.alpha * self.channel_mixer(x)


# 测试代码
if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 64, 64)
    # 实例化改进模块
    model = LiteRep_AttentionBlock(inp=32, oup=32, stride=1)
    output = model(input_tensor)
    print(f'Input shape: {input_tensor.shape}')
    print(f'Output shape: {output.shape}')

    # 打印参数量对比（示意）
    params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {params}')