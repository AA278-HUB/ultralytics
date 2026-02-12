# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = (
    "RepHDW",

)

from ultralytics.nn.modules import DWConv


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


# 先保留原有基础模块（Conv、UniRepLKNetBlock 等不变，这里假设已定义）
# ...（Conv, UniRepLKNetBlock, DilatedReparamBlock 等保持原样）

# 新增：ShiftwiseConv 模块（基于 CVPR 2025 ShiftwiseConv 核心思想）
# 简易重参数化实现：训练时 3x3 DW + channel shift，部署时合并为等效大核（可进一步优化）
class ShiftwiseConv(nn.Module):
    def __init__(self, channels, kernel_size=3, shift_range=2):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.shift_range = shift_range  # shift 步长，控制等效感受野（类似论文中 shift=2 可模拟 ~7x7）

    def forward(self, x):
        out = self.dwconv(x)
        # Channel-wise shift（上下左右循环移位，模拟大核偏移聚合）
        shifted = []
        for dx in range(-self.shift_range, self.shift_range + 1):
            for dy in range(-self.shift_range, self.shift_range + 1):
                if dx == 0 and dy == 0:
                    continue
                shifted.append(torch.roll(out, shifts=(dx, dy), dims=(2, 3)))
        if shifted:
            out = out + sum(shifted) / len(shifted)
        return out

    # 重参数化（部署时可合并 shift 为大核权重，类似 UniRepLKNet）
    def reparameterize(self):
        # 简化版：实际可通过 unfold+fold 合并为大核（参考原论文 GitHub）
        pass


# 新瓶颈：混合大核 + ShiftwiseConv + Gated
class DepthBottleneckPlus(nn.Module):
    def __init__(self, in_channels, out_channels, kersize=5, expansion_depth=2):
        super().__init__()
        mid = int(in_channels * expansion_depth)

        self.conv1 = Conv(in_channels, mid, 1)
        # 第一次：保持真大核（UniRepLKNet，强感受野）
        self.large_kernel = UniRepLKNetBlock(mid, kernel_size=kersize)
        self.conv_mid = Conv(mid, mid, 1)

        # 第二次：ShiftwiseConv（高效小核模拟大核）
        self.shift_conv = ShiftwiseConv(mid, shift_range=2)  # shift_range 可调，等效更大核

        self.conv_out = Conv(mid, out_channels, 1)
        self.act = nn.SiLU()

        # 轻量 Gate（通道注意力，生成动态权重 α）
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, mid // 8, 1),
            nn.SiLU(),
            nn.Conv2d(mid // 8, mid, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.large_kernel(y))
        y = self.conv_mid(y)

        # Shiftwise 部分 + Gate 动态加权（类似 GURLKNet gated）
        shifted = self.shift_conv(y)
        alpha = self.gate(y)  # (B, mid, 1, 1)
        y = y + alpha * shifted  # 动态调整 shift 贡献

        y = self.act(y)
        y = self.conv_out(y)
        return y


# 主块：RepHMSPlus（添加动态权重分配 + concat 前注意力）
class RepHMSPlus(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5,
                 expansion=0.5, use_depthwise=True):
        super().__init__()
        self.width = width
        self.depth = depth
        c_ = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, c_ * width, 1, 1)

        # 使用新瓶颈
        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            block_list = nn.ModuleList([
                DepthBottleneckPlus(c_, c_, kersize, depth_expansion)
                for _ in range(depth)
            ])
            self.blocks.append(block_list)

        # concat 前通道注意力（类似 YOLOv11 PSA/SE）
        self.concat_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_ * (1 + (width - 1) * depth), c_ * (1 + (width - 1) * depth) // 8, 1),
            nn.SiLU(),
            nn.Conv2d(c_ * (1 + (width - 1) * depth) // 8, c_ * (1 + (width - 1) * depth), 1),
            nn.Sigmoid()
        )

        self.conv2 = Conv(c_ * (1 + (width - 1) * depth), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * x.shape[1] // self.width:(i + 1) * x.shape[1] // self.width] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]  # 初始残差

        cascade = []
        elan = [x_out[0]]

        for i in range(self.width - 1):
            for j in range(self.depth):
                current = x_out[i + 1]
                if i > 0 and cascade:  # 动态权重分配
                    alpha = current.mean(dim=(2, 3), keepdim=True).sigmoid()  # 简易 scalar alpha（可换更强 gate）
                    current = current + alpha * cascade[j]
                    if j == self.depth - 1 and self.depth > 1:
                        cascade = [cascade[-1]]
                    elif j == self.depth - 1:
                        cascade = []
                current = self.blocks[i][j](current)
                x_out[i + 1] = current
                elan.append(current)
                if i < self.width - 2:
                    cascade.append(current)

        y = torch.cat(elan, 1)
        y = y * self.concat_attn(y)  # concat 前注意力加权
        y = self.conv2(y)
        return y


# --- 新增：Coordinate Attention (坐标注意力) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


# --- 改进：多尺度门控瓶颈 (MS-Gated Bottleneck) ---
class DepthBottleneckPro(nn.Module):
    def __init__(self, in_channels, out_channels, kersize=7, expansion_depth=2):
        super().__init__()
        mid = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid, 1)

        # 分支 1: UniRepLK 大核
        self.large_kernel = UniRepLKNetBlock(mid // 2, kernel_size=kersize)
        # 分支 2: 多尺度扩张核 (3x3 with dilation)
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(mid // 2, mid // 2, kernel_size=3, padding=2, dilation=2, groups=mid // 2),
            nn.BatchNorm2d(mid // 2),
            nn.SiLU()
        )

        self.conv_mid = Conv(mid, mid, 1)
        self.shift_conv = ShiftwiseConv(mid, shift_range=2)

        # 引入坐标注意力替代简单 Gate
        self.ca = CoordAtt(mid, mid)

        self.conv_out = Conv(mid, out_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.conv1(x))

        # 多尺度特征提取
        c1, c2 = torch.chunk(x, 2, dim=1)
        c1 = self.large_kernel(c1)
        c2 = self.dilated_conv(c2)
        x = torch.cat([c1, c2], dim=1)

        x = self.conv_mid(x)

        # Shiftwise + CA 动态加权
        shifted = self.shift_conv(x)
        x = x + self.ca(shifted)  # 使用 CA 增强空间感知

        x = self.act(x)
        x = self.conv_out(x)
        return x


# --- 主模块：RepHMSPro ---
class RepHMS_Gemini(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=7,
                 expansion=0.5):
        super().__init__()
        self.width = width
        self.depth = depth
        c_ = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, c_ * width, 1, 1)

        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            block_list = nn.ModuleList([
                DepthBottleneckPro(c_, c_, kersize, depth_expansion)
                for _ in range(depth)
            ])
            self.blocks.append(block_list)

        # 改进：在最后聚合处使用更强的 CA 注意力
        total_c = c_ * (1 + (width - 1) * depth)
        self.final_ca = CoordAtt(total_c, total_c)
        self.conv2 = Conv(total_c, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        # 灵活分块处理
        split_size = x.shape[1] // self.width
        x_out = [x[:, i * split_size: (i + 1) * split_size] for i in range(self.width)]

        x_out[1] = x_out[1] + x_out[0]

        cascade = []
        elan = [x_out[0]]

        for i in range(self.width - 1):
            for j in range(self.depth):
                current = x_out[i + 1]
                if i > 0 and cascade:
                    # 引入可学习的动态融合因子
                    current = current + cascade[j]

                current = self.blocks[i][j](current)
                x_out[i + 1] = current
                elan.append(current)

                if i < self.width - 2:
                    cascade.append(current)

        y = torch.cat(elan, 1)
        y = self.final_ca(y)  # 全局坐标注意力增强
        y = self.conv2(y)
        return y


# 新增：CBAM注意力模块（通道 + 空间注意力）
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):  # reduction越小，参数越多，但精度可能更高
        super().__init__()
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # 使用7x7卷积捕捉空间关系
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ch_weight = self.channel_attn(x)
        x = x * ch_weight
        # 空间注意力（基于max和mean池化）
        sp_input = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        sp_weight = self.spatial_attn(sp_input)
        x = x * sp_weight
        return x


# 改进瓶颈：DepthBottleneckPlusV2（升级gate为CBAM，多尺度shift）
class DepthBottleneckPlusV2(nn.Module):
    def __init__(self, in_channels, out_channels, kersize=5, expansion_depth=2):
        super().__init__()
        mid = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid, 1)
        # 保持大核
        self.large_kernel = UniRepLKNetBlock(mid, kernel_size=kersize)
        self.conv_mid = Conv(mid, mid, 1)
        # 增强ShiftwiseConv：添加多尺度shift（小/大范围并融合）
        self.shift_conv_small = ShiftwiseConv(mid, shift_range=1)  # 等效~5x5
        self.shift_conv_large = ShiftwiseConv(mid, shift_range=2)  # 等效~7x7
        self.shift_fuse = Conv(mid * 2, mid, 1)  # 融合多尺度shift
        self.conv_out = Conv(mid, out_channels, 1)
        self.act = nn.SiLU()
        # 升级gate为CBAM（通道+空间）
        self.gate = CBAM(mid)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.large_kernel(y))
        y = self.conv_mid(y)
        # 多尺度shift + 融合
        shifted_small = self.shift_conv_small(y)
        shifted_large = self.shift_conv_large(y)
        shifted = self.shift_fuse(torch.cat([shifted_small, shifted_large], dim=1))
        # 融合后应用CBAM增强
        y = y + shifted
        y = self.gate(y)  # CBAM直接增强融合特征
        y = self.act(y)
        y = self.conv_out(y)
        return y


# 改进主块：RepHMSPlusV2（使用新瓶颈，升级concat_attn为CBAM，优化动态alpha）
class RepHMSPlus_Grok(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5,
                 expansion=0.5, use_depthwise=True):
        super().__init__()
        self.width = width
        self.depth = depth
        c_ = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, c_ * width, 1, 1)
        # 使用改进瓶颈
        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            block_list = nn.ModuleList([
                DepthBottleneckPlusV2(c_, c_, kersize, depth_expansion)
                for _ in range(depth)
            ])
            self.blocks.append(block_list)
        # 升级concat_attn为CBAM（通道+空间）
        num_elan_channels = c_ * (1 + (width - 1) * depth)
        self.concat_attn = CBAM(num_elan_channels)
        self.conv2 = Conv(num_elan_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * x.shape[1] // self.width:(i + 1) * x.shape[1] // self.width] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]  # 初始残差
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                current = x_out[i + 1]
                if i > 0 and cascade:
                    # 升级动态alpha：通道-wise（使用轻量1x1 Conv + Sigmoid）
                    alpha = nn.Conv2d(current.shape[1], current.shape[1], 1)(
                        current.mean(dim=(2, 3), keepdim=True)).sigmoid()
                    current = current + alpha * cascade[j]
                    if j == self.depth - 1 and self.depth > 1:
                        cascade = [cascade[-1]]
                    elif j == self.depth - 1:
                        cascade = []
                current = self.blocks[i][j](current)
                x_out[i + 1] = current
                elan.append(current)
                if i < self.width - 2:
                    cascade.append(current)
        y = torch.cat(elan, 1)
        y = self.concat_attn(y)  # CBAM增强concat特征
        y = self.conv2(y)
        return y


import torch
import torch.nn as nn


# --- 新增：Star Operation 核心模块 ---
class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop=0.):
        super().__init__()
        # 分支 1：线性变换
        self.f1 = nn.Conv2d(dim, dim, 1, bias=True)
        # 分支 2：线性变换
        self.f2 = nn.Conv2d(dim, dim, 1, bias=True)
        # 深度卷积增强空间联系
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)

        self.act = nn.SiLU()
        self.out_conv = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x):
        # 核心：Star Operation (f1(x) * f2(x))
        # 这种乘法操作能产生类似 Attention 的高阶交互
        star = self.f1(x) * self.dw(self.f2(x))
        return self.out_conv(self.act(star))


# --- 改进：Star-Gated Bottleneck ---
class DepthBottleneckStar(nn.Module):
    def __init__(self, in_channels, out_channels, kersize=7, expansion_depth=2):
        super().__init__()
        mid = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid, 1)

        # 引入 StarBlock 进行特征高阶增强
        self.star = StarBlock(mid)

        # 保持大核感知（重参数化分支）
        self.large_kernel = UniRepLKNetBlock(mid, kernel_size=kersize)

        # 坐标注意力 (保留 Pro 版本的空间感知)
        self.ca = CoordAtt(mid, mid)

        self.conv_out = Conv(mid, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)

        # 1. 提取大核特征
        lk_feat = self.large_kernel(x)

        # 2. 注入 Star 交互特征
        # 通过 StarBlock 产生强大的非线性表达，再与大核特征相加
        x = lk_feat + self.star(x)

        # 3. 空间位置加权
        x = self.ca(x)

        x = self.conv_out(x)
        return x


# --- 主模块：RepHMSStar ---
class RepHMSStar_Gemini(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=7,
                 expansion=0.5):
        super().__init__()
        self.width = width
        self.depth = depth
        c_ = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, c_ * width, 1, 1)

        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            block_list = nn.ModuleList([
                DepthBottleneckStar(c_, c_, kersize, depth_expansion)
                for _ in range(depth)
            ])
            self.blocks.append(block_list)

        total_c = c_ * (1 + (width - 1) * depth)
        self.final_ca = CoordAtt(total_c, total_c)
        self.conv2 = Conv(total_c, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        split_size = x.shape[1] // self.width
        x_out = [x[:, i * split_size: (i + 1) * split_size] for i in range(self.width)]

        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]

        for i in range(self.width - 1):
            for j in range(self.depth):
                current = x_out[i + 1]
                if i > 0 and cascade:
                    current = current + cascade[j]

                current = self.blocks[i][j](current)
                x_out[i + 1] = current
                elan.append(current)

                if i < self.width - 2:
                    cascade.append(current)

        y = torch.cat(elan, 1)
        y = self.final_ca(y)
        y = self.conv2(y)
        return y


import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 新增：GRN (全局响应归一化) ---
# 显著提升特征通道间的竞争，增强表征能力
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # 计算 L2 范数作为响应强度
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


# --- 核心：Star-GLU 瓶颈层 ---
class OmniBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kersize=7, expansion=2.0):
        super().__init__()
        mid = int(in_channels * expansion)

        # 降维投影
        self.conv_in = Conv(in_channels, mid, 1)

        # 两个 Star 分支，模拟门控机制
        # 分支 A: 深度卷积捕捉空间结构
        self.dw_a = nn.Conv2d(mid // 2, mid // 2, kernel_size=kersize,
                              padding=kersize // 2, groups=mid // 2)
        # 分支 B: 1x1 卷积捕捉通道相关性
        self.pw_b = nn.Conv2d(mid // 2, mid // 2, 1)

        # 特征增强层
        self.grn = GRN(mid // 2)
        self.act = nn.SiLU()

        # 输出投影
        self.conv_out = Conv(mid // 2, out_channels, 1)

    def forward(self, x):
        x = self.conv_in(x)

        # Star-GLU 操作：将通道一分为二
        x1, x2 = torch.chunk(x, 2, dim=1)

        # 核心交互：(DW(x1) * PW(x2)) 产生高阶非线性
        # 加上 GRN 抑制冗余并增强对比度
        star_gate = self.dw_a(x1) * self.pw_b(x2)
        star_gate = self.grn(star_gate)

        y = self.act(star_gate)
        return self.conv_out(y)


# --- 主模块：RepHMS-Omni ---
class RepHMSOmni(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=7, expansion=0.5):
        super().__init__()
        self.width = width
        self.depth = depth
        c_ = int(out_channels * expansion)

        # 初始分流
        self.conv1 = Conv(in_channels, c_ * width, 1, 1)

        # Omni 模块组
        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            block_list = nn.ModuleList([
                OmniBottleneck(c_, c_, kersize=kersize)
                for _ in range(depth)
            ])
            self.blocks.append(block_list)

        # 引入轻量级坐标注意力
        total_c = c_ * (1 + (width - 1) * depth)
        self.ca = CoordAtt(total_c, total_c)
        self.conv2 = Conv(total_c, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        split_size = x.shape[1] // self.width
        x_out = [x[:, i * split_size: (i + 1) * split_size] for i in range(self.width)]

        # ELAN 结构的残差链
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]

        for i in range(self.width - 1):
            for j in range(self.depth):
                current = x_out[i + 1]
                if i > 0 and cascade:
                    current = current + cascade[j]

                # 经过全能感知瓶颈
                current = self.blocks[i][j](current)
                x_out[i + 1] = current
                elan.append(current)

                if i < self.width - 2:
                    cascade.append(current)

        y = torch.cat(elan, 1)
        y = self.ca(y)
        y = self.conv2(y)
        return y
# 测试入口
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 引入更强的 Dynamic Gated Large Kernel (DGLK)
class DynamicGatedLK(nn.Module):
    """
    改进点：引入双重门控机制，分别控制空间特征聚合与通道上下文。
    """

    def __init__(self, dim, kersize=7):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)  # 拆分为 Value 和 Gate
        self.dwconv = UniRepLKNetBlock(dim, kernel_size=kersize)  # 空间聚合
        self.proj_out = nn.Conv2d(dim, dim, 1)

        # 局部通道注意力，增强对跨尺度融合后的特征选择
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 分离特征路径与门控路径 (Gated Linear Unit 变体)
        u, v = self.proj_in(x).chunk(2, dim=1)
        v = self.dwconv(v)
        x = u * v  # 逐元素交互
        x = x * self.channel_gate(x)  # 通道加权
        return self.proj_out(x)


# 2. 改进版 Bottleneck：RepProBottleneck
class RepProBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kersize=5, expansion=2.0):
        super().__init__()
        mid = int(in_channels * expansion)
        self.conv1 = Conv(in_channels, mid, 1)

        # 核心：将固定 Shift 升级为 Dynamic Gated LK
        self.dglk = DynamicGatedLK(mid, kersize=kersize)

        # 引入轻量级 Shiftwise (保留其低功耗大感受野优势)
        self.shift = ShiftwiseConv(mid, shift_range=1)

        self.conv2 = Conv(mid, out_channels, 1)
        self.drop_path = nn.Identity()  # 可选 DropPath

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        # 并行/串行混合：DGLK 捕获全局，Shift 增强局部细节
        x = x + self.dglk(x)
        x = self.shift(x)
        x = self.conv2(x)
        return shortcut + x if hasattr(self, 'shortcut') else x


# 3. 终极主块：RepHMSPro
class RepHMS_Gemini_V2(nn.Module):
    """
    RepHMSPro 改进点：
    1. Weighted Cascade: 不再是简单的 alpha * cascade，而是 Cross-Gate 融合。
    2. Partial Split: 借鉴 YOLOv11 的思想，对通道进行更细致的切分以降低计算量并提升鲁棒性。
    """

    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=5, expansion=0.5):
        super().__init__()
        self.width = width
        self.c_ = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, self.c_ * width, 1)

        # 构建 blocks
        self.blocks = nn.ModuleList([
            nn.ModuleList([RepProBottleneck(self.c_, self.c_, kersize) for _ in range(depth)])
            for _ in range(width - 1)
        ])

        # 增强型多尺度融合层 (Fusion Attention)
        combined_c = self.c_ * (1 + (width - 1) * depth)
        self.fusion_attn = nn.Sequential(
            nn.Conv2d(combined_c, combined_c, 3, padding=1, groups=combined_c),  # 深度卷积增强空间感知
            nn.BatchNorm2d(combined_c),
            nn.SiLU()
        )

        self.conv2 = Conv(combined_c, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        # Split features
        x_splits = list(x.chunk(self.width, dim=1))

        elan_outputs = [x_splits[0]]
        cascade_buffer = []

        # 第一路径与第二路径初始融合
        x_splits[1] = x_splits[1] + x_splits[0]

        for i in range(self.width - 1):
            curr_feat = x_splits[i + 1]

            # 改进的 Cascade 逻辑
            if i > 0 and cascade_buffer:
                # 使用门控机制融合 cascade
                for j in range(len(cascade_buffer)):
                    # 动态权重：基于当前特征对历史特征进行选择
                    gate = torch.sigmoid(curr_feat.mean((2, 3), keepdim=True))
                    curr_feat = curr_feat + gate * cascade_buffer[j]

            for block in self.blocks[i]:
                curr_feat = block(curr_feat)
                elan_outputs.append(curr_feat)
                # 存入 cascade
                if i < self.width - 2:
                    cascade_buffer.append(curr_feat)

            # 更新下一轮输入的 cascade (仅保留最新深度特征)
            if len(cascade_buffer) > len(self.blocks[i]):
                cascade_buffer = cascade_buffer[-len(self.blocks[i]):]

        # 特征拼接与注意力增强
        out = torch.cat(elan_outputs, dim=1)
        out = self.fusion_attn(out)
        return self.conv2(out)


import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 基础轻量化组件 ---

class StarReLU(nn.Module):
    """ StarReLU: ReLU(x)^2 * s + b.
    比 SiLU 更轻量，在某些视觉任务中表现更好。
    """

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1) * scale_value)
        self.bias = nn.Parameter(torch.zeros(1) * bias_value)

    def forward(self, x):
        return self.relu(x) ** 2 * self.scale + self.bias


class StarOperation(nn.Module):
    """
    CVPR 2024 StarNet 核心思想：利用元素乘法实现高维度特征映射。
    y = (W1x + b1) * (W2x + b2)
    """

    def __init__(self, dim):
        super().__init__()
        self.f1 = nn.Conv2d(dim, dim, 1, groups=dim, bias=True)
        self.f2 = nn.Conv2d(dim, dim, 1, groups=dim, bias=True)

    def forward(self, x):
        return self.f1(x) * self.f2(x)


# --- 改进后的瓶颈模块 ---

class StarBottleneck(nn.Module):
    """
    融合了 UniRepLKNet 的大核感受野与 StarNet 的非线性交互。
    参数量较之前的 Plus 版本更低，因为减少了冗余的 1x1 卷积和 Gate 结构。
    """

    def __init__(self, in_channels, out_channels, kersize=5, expansion_depth=1.5):
        super().__init__()
        mid = int(in_channels * expansion_depth)

        # 降维投影
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid)
        )

        # 大核深度卷积 (感受野提取)
        self.large_kernel = UniRepLKNetBlock(mid, kernel_size=kersize)

        # Star-Operation (非线性特征增强，替代复杂的注意力机制)
        self.star = StarOperation(mid)

        # 激活函数统一换成轻量高效的 StarReLU
        self.act = StarReLU()

        # 最终映射
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        y = self.conv1(x)
        y = self.large_kernel(y)
        y = self.star(y)  # 元素级交互
        y = self.act(y)
        y = self.conv_out(y)
        # 如果维度一致则加残差
        return y + identity if identity.shape == y.shape else y


# --- 主块：RepHMS_Star ---

class RepHMS_Gemini_LightStar(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=1.5, kersize=5,
                 expansion=0.5):
        super().__init__()
        self.width = width
        self.depth = depth
        c_ = int(out_channels * expansion)

        # 输入切割卷积
        self.conv1 = Conv(in_channels, c_ * width, 1, 1)

        # 核心 Block 列表
        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            block_list = nn.ModuleList([
                StarBottleneck(c_, c_, kersize, depth_expansion)
                for _ in range(depth)
            ])
            self.blocks.append(block_list)

        # 融合后的 Star-Attention (利用 Star-Operation 思路简化 PSA)
        total_mid_c = c_ * (1 + (width - 1) * depth)
        self.fusion_star = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_mid_c, total_mid_c, 1, groups=total_mid_c),  # Star 风格的 Scale
            nn.Sigmoid()
        )

        self.conv2 = Conv(total_mid_c, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        # Split
        chunks = torch.chunk(x, self.width, dim=1)
        x_out = list(chunks)

        # 初始级联
        x_out[1] = x_out[1] + x_out[0]

        elan = [x_out[0]]
        cascade = []

        for i in range(self.width - 1):
            current = x_out[i + 1]
            for j in range(self.depth):
                # 级联路径优化：使用简单的乘法注入
                if i > 0 and cascade:
                    current = current + cascade[j]

                current = self.blocks[i][j](current)
                elan.append(current)

                if i < self.width - 2:
                    cascade.append(current)
            # 更新下一路输入
            if i + 2 < len(x_out):
                x_out[i + 2] = x_out[i + 2] + current

        y = torch.cat(elan, 1)
        # 融合增强
        y = y * self.fusion_star(y)
        y = self.conv2(y)
        return y
if __name__ == "__main__":
    # 创建模型（输入通道64，输出通道256，常见 YOLO 配置）
    model = RepHMSPlus(in_channels=64, out_channels=256, width=3, depth=1, kersize=5, expansion=0.5)

    # 随机输入（B, C, H, W）
    x = torch.randn(1, 64, 64, 64)

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("RepHMSPlus 测试通过！模型可以正常运行。")