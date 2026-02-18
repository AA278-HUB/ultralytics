import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2f, Conv
from ultralytics.nn.modules.block import C3k, C3
from ultralytics.nn.modules.conv import autopad


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def get_bn(channels):
    return nn.BatchNorm2d(channels)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""

    def __init__(self, c1, ratio=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // ratio, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RepConv(nn.Module):
    """
    RepVGG-style Convolution Block.
    Training: 3x3 Conv + 1x1 Conv + Identity (if applicable).
    Inference: Fused into a single 3x3 Conv.
    用于大幅提升局部特征提取能力并增加参数量。
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = k
        self.stride = s
        self.padding = k // 2 if p is None else p
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, self.padding, groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(c1) if c2 == c1 and s == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, self.padding, groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None: return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'): return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                                     groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters(): para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'): self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'): self.__delattr__('id_tensor')
        self.deploy = True


class UniRepLK_Bottleneck(nn.Module):
    """使用大核重参数化设计的瓶颈模块，用于 C3k。"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=7, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # 1. 局部特征提取 (RepConv 3x3)
        self.cv1 = RepConv(c1, c_, k=3, s=1, g=g)
        # 2. 全局感受野 (Large Kernel DW Conv)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, k, 1, k // 2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )
        # 3. 通道注意力
        self.attn = SEBlock(c_, ratio=8)
        # 4. 投影输出 1x1
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.attn(self.cv2(self.cv1(x)))) if self.add else self.cv3(self.attn(self.cv2(self.cv1(x))))

class C3k_UniRepLK(C3):
    """C3k 模块的 UniRepLK 版本，内部嵌套多个 UniRepLK_Bottleneck。"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, k=7, e=0.5,):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # 将原有的 Bottleneck 替换为 UniRepLK 风格的 Bottleneck
        self.m = nn.Sequential(*(UniRepLK_Bottleneck(c_, c_, shortcut, g, k=k, e=1.0) for _ in range(n)))



# class C3k2(C2f):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
#
#     def __init__(
#         self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
#     ):
#         """
#         Initialize C3k2 module.
#
#         Args:
#             c1 (int): Input channels.
#             c2 (int): Output channels.
#             n (int): Number of blocks.
#             c3k (bool): Whether to use C3k blocks.
#             e (float): Expansion ratio.
#             g (int): Groups for convolutions.
#             shortcut (bool): Whether to use shortcut connections.
#         """
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
#         )

class C3k2_UniRepLKv2(C2f):
    """
    C3k2 的 UniRepLK 进阶版。
    - c3k=True: 内部嵌套 C3k_UniRepLK (更深, 两次 CSP 结构)
    - c3k=False: 内部直接使用 UniRepLK_Bottleneck (较浅, 但保留大核和重参数化)
    """

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, g=1, shortcut=True, k=7):
        super().__init__(c1, c2, n, shortcut, g, e)

        # 统一使用 UniRepLK 系列组件
        self.m = nn.ModuleList(
            C3k_UniRepLK(self.c, self.c, 2, shortcut, g, k=k) if c3k else
            UniRepLK_Bottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )