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

# class BiFPN_SumX(nn.Module):
# #     def __init__(self, num_inputs, channels=512, act='swish'):
# #         super(BiFPN_SumX, self).__init__()  # 修正 super 调用
# #         # 可学习权重 shape: [num_inputs, channels] → 总参数量 = num_inputs * channels
# #         self.w = nn.Parameter(torch.ones(num_inputs, channels, dtype=torch.float32), requires_grad=True)
# #         self.channels = channels
# #         self.act = nn.SiLU() if act == 'swish' else nn.ReLU()
# #
# #     def forward(self, x):  # x: list[Tensor], 每个 [channels, H, W]
# #         # per-channel softmax：在 num_inputs 维度上独立 softmax
# #         weight = torch.softmax(self.w, dim=0)  # [num_inputs, channels]
# #
# #         # 向量化加权求和
# #         x_stack = torch.stack(x, dim=0)  # [num_inputs, channels, H, W]
# #         weight = weight.view(-1, self.channels, 1, 1)  # [num_inputs, channels, 1, 1]
# #         out = (weight * x_stack).sum(dim=0)  # [channels, H, W]
# #
# #         return self.act(out)  # post-fusion 激活
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_SumX(nn.Module):
    def __init__(self, num_inputs, channels=512, act='swish'):
        super(BiFPN_SumX, self).__init__()
        # 可学习权重: [num_inputs, channels]
        self.w = nn.Parameter(torch.ones(num_inputs, channels, dtype=torch.float32), requires_grad=True)
        self.num_inputs = num_inputs
        self.channels = channels
        self.act = nn.SiLU() if act == 'swish' else nn.ReLU()

    def forward(self, x):  # x: list[Tensor], 每个 shape [B, channels, H, W]（B、H、W 必须相同）
        # per-channel softmax：在 num_inputs 维度独立 softmax
        weight = torch.softmax(self.w, dim=0)  # [num_inputs, channels]

        # 添加 batch 维度以正确广播
        weight = weight.unsqueeze(1)  # [num_inputs, 1, channels]
        weight = weight.unsqueeze(-1).unsqueeze(-1)  # [num_inputs, 1, channels, 1, 1]

        # 堆叠输入
        x_stack = torch.stack(x, dim=0)  # [num_inputs, B, channels, H, W]

        # 加权求和
        out = (x_stack * weight).sum(dim=0)  # [B, channels, H, W]

        return self.act(out)