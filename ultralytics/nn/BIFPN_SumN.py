import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN_SumN(nn.Module):
    def __init__(self, num_inputs, act='swish'):  # 添加out_channels和激活
        super(BiFPN_SumN, self).__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.act = nn.SiLU() if act == 'swish' else nn.ReLU()  # BiFPN常用Swish

    def forward(self, x):  # x: list[Tensor], 每个已调整到相同C,H,W
        w = F.relu(self.w)
        weight = w / (torch.sum(w) + self.epsilon)
        out = sum(weight[i] * x[i] for i in range(len(x)))
        return self.act(out)  # post-fusion激活，提升表达