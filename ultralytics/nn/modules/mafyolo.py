# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .conv import DWConv


__all__ = (
    "RepHDW",

)

class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class AVG(nn.Module):
    def __init__(self, down_n=2):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.down_n = down_n
        # self.output_size = np.array([H, W])

    def forward(self, x):
        B, C, H, W = x.shape
        H = int(H / self.down_n)
        W = int(W / self.down_n)
        output_size = np.array([H, W])
        x = self.avg_pool(x, output_size)
        return x

class RepHDW(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut = True, expansion = 0.5, kersize = 5,depth_expansion = 1,small_kersize = 3,use_depthwise = True):
        super(RepHDW, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(DepthBottleneckUni(self.c_, self.c_, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.conv2 = Conv(c_ * (depth+2), out_channels, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return  y_out

class DepthBottleneckUni(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize = 5,
                 expansion_depth = 1,
                 small_kersize = 3,
                 use_depthwise=True):
        super(DepthBottleneckUni, self).__init__()


        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:

            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel,out_channels,kernel_size = 1)

        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)

        y = self.act(self.conv2(y))

        y = self.one_conv(y)
        return y

class UniRepLKNetBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 deploy=False,
                 attempt_use_lk_impl=True):
        super().__init__()
        if deploy:
            print('------------------------------- Note: deploy mode')
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in [3]
            self.dwconv = get_conv2d_uni(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=deploy,
                                     attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim)


    def forward(self, inputs):

        out = self.norm(self.dwconv(inputs))
        return out

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                            self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 self.dwconv.padding, self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d_uni(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [7, 5, 3]
            self.dilates = [1, 1, 1]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 1]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        elif kernel_size == 3:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]


        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_uni(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
def get_conv2d_uni(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)
def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device =kernel.device )
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel
def get_bn(channels):
    return nn.BatchNorm2d(channels)

class DepthBottleneckUniv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckUniv2, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = UniRepLKNetBlock(mid_channel2, kernel_size=kersize)
            self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        y = self.act1(self.conv3(y))
        y = self.one_conv2(y)
        return y

class RepHMS(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):
        super(RepHMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUniv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        #cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade =[cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


class DepthBottleneckv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckv2, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = DWConv(mid_channel, mid_channel, kersize)
            # self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = DWConv(mid_channel2, mid_channel2, kersize)
            # self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.one_conv(y)
        y = self.conv3(y)
        y = self.one_conv2(y)
        return y


class ConvMS(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):
        super(ConvMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        # cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out
    

# 假设已有 Conv, UniRepLKNetBlock, get_bn 等原模块

class GatedAttention(nn.Module):  # 新增：Gated Attention (受  启发)
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        gate = self.act(self.fc2(F.relu(self.fc1(avg_pool))))
        return x * gate.unsqueeze(-1).unsqueeze(-1)

class DepthBottleneckUniv3(nn.Module):  # 升级瓶颈块：更大内核 + 重参数化 + 注意
    def __init__(self, in_channels, out_channels, shortcut=True, kersize=7,  # 默认更大内核
                 expansion_depth=2, small_kersize=3, use_depthwise=True, reparam=True):
        super().__init__()
        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)  # 更大内核
            self.act = nn.SiLU()
            if reparam:  # 非对称重参数化
                self.reparam_branch = nn.Sequential(
                    Conv(mid_channel, mid_channel, 1),  # 1x1 分支
                    Conv(mid_channel, mid_channel, 3, 1, groups=mid_channel)  # 3x3 DW 分支
                )
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)
            self.conv3 = UniRepLKNetBlock(mid_channel2, kernel_size=kersize)
            self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
            self.attention = GatedAttention(out_channels)  # 新增注意
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        if hasattr(self, 'reparam_branch'):  # 重参数化应用
            y += self.reparam_branch(y)  # 训练时添加，推理时合并
        y = self.one_conv(y)
        y = self.act1(self.conv3(y))
        y = self.one_conv2(y)
        y = self.attention(y)  # 应用 gated attention
        if self.shortcut:
            y += x  # 残差连接
        return y

    def reparameterize(self):  # 推理时合并
        if hasattr(self, 'reparam_branch'):
            # 合并 reparam_branch 到 conv2/conv3 (伪代码，实际需实现 fuse_bn 等)
            self.conv2.weight.data += self.reparam_branch[1].weight.data  # 示例合并
            del self.reparam_branch

#Grok 帮写
class RepHMSv2(nn.Module):  # 主模块：融入 R-ELAN 风格融合
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2,
                 kersize=7, shortcut=True, expansion=0.5, small_kersize=3,
                 use_depthwise=True, reparam=True):  # 新增 reparam
        super().__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUniv3(self.c_, self.c_, shortcut, kersize, depth_expansion,
                                     small_kersize, use_depthwise, reparam)  # 使用新瓶颈
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)
        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)
        self.residual_agg = nn.ModuleList([nn.Identity() for _ in range(depth)])  # R-ELAN 残差聚合

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]  # 初始融合
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j] + self.residual_agg[j](x_out[0])  # 添加 R-ELAN 残差
                    if j == self.depth - 1:
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])
        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out



#=============Gemini帮写==============
class SimpleGate(nn.Module):
    """一个轻量级的门控模块，用于过滤级联时的冗余信息"""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        return x + residual * self.gate(x)

'''1'''
class RepGMS(nn.Module):
    """
    RepGMS: Reparameterized Gated Multi-Scale Block
    在 RepHMS 基础上增加了 Gate 融合机制和特征校准
    """

    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5, small_kersize=3, use_depthwise=True):
        super(RepGMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_

        self.conv1 = Conv(in_channels, c1, 1, 1)

        # 改进 1: 引入门控融合，减少级联噪声
        self.gates = nn.ModuleList([SimpleGate(c_) for _ in range(width - 1)])

        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUniv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        # 改进 2: 增加一个最终的特征校准层 (类似 SE 但更轻量)
        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        # Split features
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]

        # 第一路径级联
        x_out[1] = x_out[1] + x_out[0]

        cascade = []
        elan = [x_out[0]]

        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    # 核心改进：使用门控融合替代直接相加
                    # x_out[i + 1] = x_out[i + 1] + cascade[j]
                    x_out[i + 1] = self.gates[i - 1](x_out[i + 1], cascade[j])

                    if j == self.depth - 1:
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []

                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])

                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out #

class StarReLU(nn.Module):
    """来自最新 Vision Transformer 研究，比 SiLU 更高效且具备更好的非线性表达"""

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.tensor(scale_value))
        self.bias = nn.Parameter(torch.tensor(bias_value))

    def forward(self, x):
        return self.scale * (self.relu(x) ** 2) + self.bias


class PConv(nn.Module):
    """Partial Convolution: 减少冗余计算，提升特征提取效率"""

    def __init__(self, dim, n_div=4, forward='split_cat', kernel_size=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, kernel_size // 2, bias=False)

    def forward(self, x):
        # 仅对一部分通道进行卷积，其余透传
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)

class GCI_Module(nn.Module):
    """Global Context Injection: 全局信息注入模块"""

    def __init__(self, channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, global_feat):
        # 将全局信息调制到当前分支
        return x * self.gate(global_feat)

class RepGVA_Bottleneck(nn.Module):
    def __init__(self, c1, c2, k=5, expansion=1.0):
        super().__init__()
        c_ = int(c2 * expansion)
        self.cv1 = Conv(c1, c_, 1, 1)
        # 结合 PConv 和 UniRepLK 的大核思想
        self.pconv = PConv(c_, n_div=4, kernel_size=k)
        self.lk_block = UniRepLKNetBlock(c_, kernel_size=k)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.act = StarReLU()

    def forward(self, x):
        y = self.cv1(x)
        y = self.pconv(y)
        y = self.lk_block(y)
        return self.cv2(self.act(y))

'''2'''
class RepGVA_ELAN(nn.Module):
    """
    最新一代 RepGVA-ELAN:
    集成了全局上下文注入、Partial Conv 以及 StarReLU
    """

    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=5, expansion=0.5):
        super().__init__()
        self.width = width
        self.c_ = int(out_channels * expansion)
        c_total = self.c_ * width

        self.conv1 = Conv(in_channels, c_total, 1, 1)

        # 全局特征提取分支
        self.global_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(self.c_, self.c_, 1, 1)
        )

        # 全局注入模块
        self.gci_modules = nn.ModuleList([GCI_Module(self.c_) for _ in range(width - 1)])

        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            self.blocks.append(nn.ModuleList([
                RepGVA_Bottleneck(self.c_, self.c_, k=kersize) for _ in range(depth)
            ]))

        self.conv2 = Conv(self.c_ * (1 + (width - 1) * depth), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = list(x.split(self.c_, 1))

        # 提取第一个分支作为全局信息源
        global_context = self.global_extractor(x_out[0])

        elan_results = [x_out[0]]
        for i in range(self.width - 1):
            branch_feat = x_out[i + 1]
            # 注入全局信息
            branch_feat = self.gci_modules[i](branch_feat, global_context)

            for j in range(len(self.blocks[i])):
                branch_feat = self.blocks[i][j](branch_feat)
                elan_results.append(branch_feat)

        return self.conv2(torch.cat(elan_results, 1))

class DynamicGate(nn.Module):
    """动态上下文门控：通过全局感受野生成空间/通道权重"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = channels // reduction
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        # 根据当前主干特征 x，动态调整跳连特征 skip 的注入强度
        g = self.gate(x)
        return x + g * skip
class PartialUniRepBottleneck(nn.Module):
    """
    结合 PConv 思想的大核瓶颈模块
    只对 1/4 的通道进行昂贵的大核卷积，其余通道保留原始特征
    """

    def __init__(self, in_channels, out_channels, kersize=5, p_ratio=0.25):
        super().__init__()
        self.split_c = int(in_channels * p_ratio)
        self.conv_p = UniRepLKNetBlock(self.split_c, kernel_size=kersize)
        self.conv_1x1 = Conv(in_channels, out_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        # 部分卷积：只处理 split_c 长度的通道
        x1, x2 = torch.split(x, [self.split_c, x.shape[1] - self.split_c], dim=1)
        x1 = self.conv_p(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.act(self.conv_1x1(x))


''''3'''
class RepDGM(nn.Module):
    """
    RepDGM: Reparameterized Dynamic Gated Multi-scale Block
    集成了：1. PConv 算子 2. 动态门控融合 3. ELAN 级联结构
    """

    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=5, expansion=0.5):
        super(RepDGM, self).__init__()
        self.width = width
        self.c_ = int(out_channels * expansion)

        # 输入投影
        self.conv1 = Conv(in_channels, self.c_ * width, 1, 1)

        # 动态门控组
        self.gates = nn.ModuleList([DynamicGate(self.c_) for _ in range(width - 1)])

        # 深度增强组
        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            self.blocks.append(nn.ModuleList([
                PartialUniRepBottleneck(self.c_, self.c_, kersize=kersize)
                for _ in range(depth)
            ]))

        # 最终融合
        self.conv2 = Conv(self.c_ * (1 + (width - 1) * depth), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        # 初始特征切分
        x_splits = list(x.split(self.c_, dim=1))

        results = [x_splits[0]]  # 存储最终 cat 的特征
        cascade_buffer = x_splits[0]  # 用于跨层传递的缓冲

        for i in range(self.width - 1):
            curr_feat = x_splits[i + 1]

            # 使用动态门控融合前一层的级联信息
            curr_feat = self.gates[i](curr_feat, cascade_buffer)

            for j in range(len(self.blocks[i])):
                curr_feat = self.blocks[i][j](curr_feat)
                results.append(curr_feat)

            # 更新级联缓冲（通常取该分支的最后输出）
            cascade_buffer = curr_feat

        y = torch.cat(results, dim=1)
        return self.conv2(y)

class DynamicGate(nn.Module):
    """动态上下文门控：通过全局感受野生成空间/通道权重"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = channels // reduction
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        # 根据当前主干特征 x，动态调整跳连特征 skip 的注入强度
        g = self.gate(x)
        return x + g * skip
class PartialUniRepBottleneck(nn.Module):
    """
    结合 PConv 思想的大核瓶颈模块
    只对 1/4 的通道进行昂贵的大核卷积，其余通道保留原始特征
    """

    def __init__(self, in_channels, out_channels, kersize=5, p_ratio=0.25):
        super().__init__()
        self.split_c = int(in_channels * p_ratio)
        self.conv_p = UniRepLKNetBlock(self.split_c, kernel_size=kersize)
        self.conv_1x1 = Conv(in_channels, out_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        # 部分卷积：只处理 split_c 长度的通道
        x1, x2 = torch.split(x, [self.split_c, x.shape[1] - self.split_c], dim=1)
        x1 = self.conv_p(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.act(self.conv_1x1(x))

'''4'''
class RepDGM_V2(nn.Module):
    """
    RepDGM: Reparameterized Dynamic Gated Multi-scale Block
    集成了：1. PConv 算子 2. 动态门控融合 3. ELAN 级联结构
    """

    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=5, expansion=0.5):
        super(RepDGM_V2, self).__init__()
        self.width = width
        self.c_ = int(out_channels * expansion)

        # 输入投影
        self.conv1 = Conv(in_channels, self.c_ * width, 1, 1)

        # 动态门控组
        self.gates = nn.ModuleList([DynamicGate(self.c_) for _ in range(width - 1)])

        # 深度增强组
        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            self.blocks.append(nn.ModuleList([
                PartialUniRepBottleneck(self.c_, self.c_, kersize=kersize)
                for _ in range(depth)
            ]))

        # 最终融合
        self.conv2 = Conv(self.c_ * (1 + (width - 1) * depth), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        # 初始特征切分
        x_splits = list(x.split(self.c_, dim=1))

        results = [x_splits[0]]  # 存储最终 cat 的特征
        cascade_buffer = x_splits[0]  # 用于跨层传递的缓冲

        for i in range(self.width - 1):
            curr_feat = x_splits[i + 1]

            # 使用动态门控融合前一层的级联信息
            curr_feat = self.gates[i](curr_feat, cascade_buffer)

            for j in range(len(self.blocks[i])):
                curr_feat = self.blocks[i][j](curr_feat)
                results.append(curr_feat)

            # 更新级联缓冲（通常取该分支的最后输出）
            cascade_buffer = curr_feat

        y = torch.cat(results, dim=1)
        return self.conv2(y)
class StarBottleneck(nn.Module):
    """
    基于 StarNet 思想的星型瓶颈模块
    利用元素级乘法模拟高阶特征交互
    """

    def __init__(self, ch, kersize=5, expansion=1.0):
        super().__init__()
        hidden_ch = int(ch * expansion)
        self.pre_conv = Conv(ch, hidden_ch * 2, 1)  # 一次性投影到两倍维度
        self.dw_conv = UniRepLKNetBlock(hidden_ch, kernel_size=kersize)
        self.post_conv = Conv(hidden_ch, ch, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        # 1. 投影并切分为两路
        x_split = self.pre_conv(x)
        x1, x2 = torch.split(x_split, x_split.shape[1] // 2, dim=1)

        # 2. 支路 1 进行大核深度卷积增强感官，支路 2 作为调制器
        x1 = self.dw_conv(x1)

        # 3. 星型运算：元素级乘法 (High-dimensional feature interaction)
        y = x1 * x2

        # 4. 投影回原始维度并加残差
        return x + self.post_conv(self.act(y))
class AdaptiveModulator(nn.Module):
    """坐标感知的自适应调制模块"""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c // 4, 1),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 4, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        # 计算空间权重，动态调制跳连特征
        weight = self.conv(x)
        return x + weight * skip
'''5'''
class RepSFA(nn.Module):
    """
    RepSFA: Reparameterized Sparse Fusion & Adaptive Modulation Block
    设计亮点：
    1. 引入 Star-Operation 提升特征映射的非线性
    2. 使用自适应调制器优化多尺度级联
    3. 结构重参数化友好
    """

    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=5, expansion=0.5):
        super(RepSFA, self).__init__()
        self.width = width
        self.c_ = int(out_channels * expansion)

        # 输入线性投影
        self.conv1 = Conv(in_channels, self.c_ * width, 1, 1)

        # 级联调制器：让融合变得“聪明”
        self.modulators = nn.ModuleList([AdaptiveModulator(self.c_) for _ in range(width - 1)])

        # 深度 Star 模块组
        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            self.blocks.append(nn.ModuleList([
                StarBottleneck(self.c_, kersize=kersize)
                for _ in range(depth)
            ]))

        self.conv2 = Conv(self.c_ * (1 + (width - 1) * depth), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_splits = list(x.split(self.c_, dim=1))

        elan_results = [x_splits[0]]
        current_cascade = x_splits[0]

        for i in range(self.width - 1):
            feat = x_splits[i + 1]

            # 改进：使用 AFM 进行特征调制融合
            feat = self.modulators[i](feat, current_cascade)

            for j in range(len(self.blocks[i])):
                feat = self.blocks[i][j](feat)
                elan_results.append(feat)

            current_cascade = feat

        return self.conv2(torch.cat(elan_results, dim=1))


class StarBottleneckPro(nn.Module):
    """
    重构后的星型交互瓶颈：利用高阶非线性提升特征提取强度
    """

    def __init__(self, ch, kersize=5, expansion=1.0):
        super().__init__()
        mid_ch = int(ch * expansion)
        # 一次性投影得到两个分支进行交互
        self.pre_conv = Conv(ch, mid_ch * 2, 1)
        self.dw_conv = UniRepLKNetBlock(mid_ch, kernel_size=kersize)
        self.post_conv = Conv(mid_ch, ch, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        # 产生两个高维分支
        mu_sigma = self.pre_conv(x)
        mu, sigma = torch.split(mu_sigma, mu_sigma.shape[1] // 2, dim=1)
        # 星型运算：x = mu * Conv(sigma)，模拟自注意力的非线性映射
        out = mu * self.dw_conv(sigma)
        return x + self.post_conv(self.act(out))


class SelectiveFusion(nn.Module):
    """
    轻量化动态选择器：决定级联信息的注入权重
    """

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 4, c, 1),
            nn.Sigmoid()
        )

    def forward(self, current, skip):
        # 根据当前特征动态调整跳连特征
        return current + self.gate(current) * skip

'''6'''
class RepHMS_Pro(nn.Module):
    """
    RepHMS-Pro: 在 RepHMS 结构基础上进行的究极进化版
    1. 级联方式由线性加和改为 Selective Fusion
    2. 核心算子由 DepthBottleneck 改为 StarBottleneckPro
    """

    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=5, expansion=0.5):
        super(RepHMS_Pro, self).__init__()
        self.width = width
        self.depth = depth
        self.c_ = int(out_channels * expansion)

        # 输入通道投影
        self.conv1 = Conv(in_channels, self.c_ * width, 1, 1)

        # 动态级联选择器组
        self.fusion_layers = nn.ModuleList([SelectiveFusion(self.c_) for _ in range(width - 1)])

        # 高阶星型模块组
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            self.RepElanMSBlock.append(nn.ModuleList([
                StarBottleneckPro(self.c_, kersize=kersize)
                for _ in range(depth)
            ]))

        # 最终特征聚合
        self.conv2 = Conv(self.c_ * (1 + (width - 1) * depth), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]

        # 第一路径简单叠加（保留基础流）
        x_out[1] = x_out[1] + x_out[0]

        cascade = []
        elan = [x_out[0]]

        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    # 核心改进：动态选择融合
                    x_out[i + 1] = self.fusion_layers[i - 1](x_out[i + 1], cascade[j])

                    if j == self.depth - 1:
                        cascade = [cascade[-1]] if self.depth > 1 else []

                # 核心改进：高阶特征提取
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])

                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        return self.conv2(torch.cat(elan, 1))


class DLKA_Bottleneck(nn.Module):
    """
    D-LKA: 动态大核注意力瓶颈
    不仅提取特征，还生成空间掩码进行自校准
    """

    def __init__(self, ch, kersize=5, expansion=1.5):
        super().__init__()
        hidden_ch = int(ch * expansion)
        self.pre_conv = Conv(ch, hidden_ch, 1)

        # 大核注意力分支：利用重参数化大核捕捉全局依赖
        self.attn_dw = UniRepLKNetBlock(hidden_ch, kernel_size=kersize)
        self.attn_1x1 = nn.Conv2d(hidden_ch, hidden_ch, 1)

        # 局部特征分支
        self.local_conv = DWConv(hidden_ch, hidden_ch, 3)

        self.post_conv = Conv(hidden_ch, ch, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.pre_conv(x)
        # 生成动态注意力掩码
        attn = self.attn_1x1(self.attn_dw(x))
        attn = torch.sigmoid(attn)
        # 注意力加权 + 局部特征补充
        x = x * attn + self.local_conv(x)
        return self.act(self.post_conv(x))


class BranchAttention(nn.Module):
    """
    分支间上下文校准模块
    让级联的信息不再是死板的相加，而是基于空间上下文的对齐
    """

    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 4, c, 1),
            nn.Sigmoid()
        )

    def forward(self, current, skip):
        # 计算全局权重
        scale = self.fc(self.pool(current + skip))
        return current + skip * scale

'''7'''
class RepHMA(nn.Module):
    """
    RepHMA: 究极进化的重参数化多尺度注意力模块
    结构上继承 RepHMS，逻辑上引入动态注意力与分支校准
    """

    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=7, expansion=0.5):
        super(RepHMA, self).__init__()
        self.width = width
        self.depth=depth
        self.c_ = int(out_channels * expansion)

        # 输入投影
        self.conv1 = Conv(in_channels, self.c_ * width, 1, 1)

        # 分支校准层
        self.calibrators = nn.ModuleList([BranchAttention(self.c_) for _ in range(width - 1)])

        # 核心 D-LKA 模块组
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            self.RepElanMSBlock.append(nn.ModuleList([
                DLKA_Bottleneck(self.c_, kersize=kersize)
                for _ in range(depth)
            ]))

        # 最终融合
        self.conv2 = Conv(self.c_ * (1 + (width - 1) * depth), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]

        # 初始化级联
        x_out[1] = x_out[1] + x_out[0]

        cascade = []
        elan = [x_out[0]]

        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    # 核心改进：分支间自适应校准融合
                    x_out[i + 1] = self.calibrators[i - 1](x_out[i + 1], cascade[j])

                    if j == self.depth - 1:
                        cascade = [cascade[-1]] if self.depth > 1 else []

                # 核心改进：D-LKA 动态特征提取
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])

                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        return self.conv2(torch.cat(elan, 1))

# if __name__ == "__main__":
#     # 测试 RepHMSv2 模块
#     model = RepHMSv2(in_channels=3, out_channels=64)
#     model.eval()  # 设置为评估模式，避免 BN 问题
#     input_tensor = torch.randn(1, 3, 224, 224)
#     output = model(input_tensor)
#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output.shape)
#     print("Test completed successfully.")