import math

import numpy as np
import torch
import torch.nn as nn


# 结合BiFPN 设置可学习参数 学习不同分支的权重
# 两个分支concat操作
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)


# import torch
# import torch.nn as nn
import torch.nn.functional as F


# BiFPN 原论文：两个分支的加权 sum 融合
class BiFPN_Sum2(nn.Module):
    def __init__(self,dimension=1):
        super(BiFPN_Sum2, self).__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x):
        """
        x: list[Tensor]，长度为 2
        每个 Tensor 形状必须一致: [B, C, H, W]
        """
        # 原论文做法：ReLU 保证权重非负
        w = F.relu(self.w)

        # Fast normalized fusion
        weight = w / (torch.sum(w) + self.epsilon)

        # 加权求和（sum，而不是 concat）
        out = weight[0] * x[0] + weight[1] * x[1]
        return out
# BiFPN 原论文：三个分支的加权 sum 融合
class BiFPN_Sum3(nn.Module,):
    def __init__(self,dimension=1):
        super(BiFPN_Sum3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x):
        """
        x: list[Tensor]，长度为 3
        每个 Tensor 形状必须一致: [B, C, H, W]
        """
        w = F.relu(self.w)

        # Fast normalized fusion
        weight = w / (torch.sum(w) + self.epsilon)

        out = (
            weight[0] * x[0] +
            weight[1] * x[1] +
            weight[2] * x[2]
        )
        return out
