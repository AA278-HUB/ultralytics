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

# ==================== 新增：Coordinate Attention (比SE强) ====================
# ==================== 修正后的 Coordinate Attention（支持任意小通道） ====================
class CoordinateAttention(nn.Module):
    """Coordinate Attention - 同时捕捉通道 + 水平/垂直空间方向信息（已修复小通道崩溃）"""

    def __init__(self, c1, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 关键修复：隐藏维度至少为1，彻底杜绝 out_channels=0
        hidden = max(16, c1 // reduction)

        self.conv1 = nn.Conv2d(c1, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, c1, kernel_size=1, bias=False)  # h方向
        self.conv3 = nn.Conv2d(hidden, c1, kernel_size=1, bias=False)  # w方向
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_h = self.conv2(y_h).sigmoid()  # [B, C, H, 1]
        y_w = self.conv3(y_w.permute(0, 1, 3, 2)).sigmoid()  # [B, C, 1, W]

        return identity * y_h * y_w


# ==================== 新增：Dilated Reparam 大核块 (UniRepLKNet风格) ====================
# ==================== 修正后的 DilatedReparamConv（自动计算padding） ====================
class DilatedReparamConv(nn.Module):
    """多分支dilated DW Conv（UniRepLKNet风格），自动计算padding，彻底避免形状不匹配"""
    def __init__(self, c, k=13):   # k只是名义上的感受野参考，可随意改
        super().__init__()
        self.c = c
        self.id_bn = nn.BatchNorm2d(c)

        # 多分支配置：(kernel_size, dilation)
        branch_configs = [
            (5, 1),   # 5x5 d=1
            (7, 2),   # 7x7 d=2
            (3, 3),   # 3x3 d=3
            (3, 4),   # 3x3 d=4
            (3, 5),   # 3x3 d=5  → 覆盖到约13的稀疏感受野
        ]

        self.branches = nn.ModuleList()
        for ks, dil in branch_configs:
            pad = dil * (ks - 1) // 2          # ← 关键：自动正确padding
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(c, c, ks, stride=1, padding=pad, dilation=dil, groups=c, bias=False),
                    nn.BatchNorm2d(c),
                    nn.SiLU()
                )
            )

    def forward(self, x):
        out = self.id_bn(x)                    # identity分支，尺寸不变
        for branch in self.branches:
            out = out + branch(x)              # 现在所有分支输出尺寸一定相同
        return out

# ==================== 增强版瓶颈 (核心替换) ====================
class EnhancedUniRepLK_Bottleneck(nn.Module):
    """升级版UniRepLK瓶颈：Dilated大核 + CA + FFN，精度提升显著"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=13, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k=3, s=1, g=g)          # 局部特征 (保持你的RepConv)
        self.cv2 = DilatedReparamConv(c_, k=k)             # 多尺度稀疏大核 (核心升级)
        self.attn = CoordinateAttention(c_)                # 空间方向注意力
        # 轻量FFN (通道增强，像UniRepLKNet的FFN)
        self.ffn = nn.Sequential(
            nn.Conv2d(c_, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
            nn.Conv2d(c_, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
        )
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x
        x = self.cv1(x)
        x = self.cv2(x)                    # 大核全局
        x = self.attn(x)                   # 空间注意力
        x = self.ffn(x) + x                # FFN残差增强
        x = self.cv3(x)
        return residual + x if self.add else x

# ==================== 升级后的C3k_UniRepLK和C3k2 ====================
class C3k_UniRepLK_Enhanced(C3):
    """C3k的增强版，使用新瓶颈"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, k=13, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(EnhancedUniRepLK_Bottleneck(c_, c_, shortcut, g, k=k, e=1.0) for _ in range(n))
        )

class C3k2_UniRepLKv3(C2f):
    """
    C3k2 的最终增强版 v3（推荐替换原来的v2）
    - c3k=True：使用更深的 C3k_UniRepLK_Enhanced（两次CSP）
    - c3k=False：直接使用Enhanced瓶颈（浅但强）
    参数冗余已充分考虑，精度提升最明显
    """
    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, g=1, shortcut=True, k=13):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_UniRepLK_Enhanced(self.c, self.c, 2, shortcut, g, k=k) if c3k else
            EnhancedUniRepLK_Bottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )


# ---------------------- 核心组件：多尺度重参数化大核 DW ----------------------
class MultiScaleRepDW(nn.Module):
    """
    多尺度重参数化深度卷积 (Multi-Scale Reparameterized Depthwise Conv).
    训练时：并行计算 3x3, 5x5, 7x7 DWConv，捕捉多尺度特征。
    推理时：融合为一个单一的 7x7 DWConv，速度极快。
    """

    def __init__(self, c1, kernel_sizes=(3, 5, 7), stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.c1 = c1
        self.stride = stride
        self.max_k = max(kernel_sizes)

        if deploy:
            # 推理模式：一个大的 DWConv
            self.dw_reparam = nn.Conv2d(c1, c1, self.max_k, stride, self.max_k // 2, groups=c1, bias=True)
        else:
            # 训练模式：多个并行的 DWConv 和 BatchNorm
            self.branches = nn.ModuleList()
            for k in kernel_sizes:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(c1, c1, k, stride, k // 2, groups=c1, bias=False),
                    nn.BatchNorm2d(c1)
                ))

            # 可选：加入 Identity 分支 (如果 stride=1)
            self.use_identity = (stride == 1)
            if self.use_identity:
                self.identity = nn.BatchNorm2d(c1)

    def forward(self, x):
        if self.deploy:
            return self.dw_reparam(x)

        out = 0
        for branch in self.branches:
            out += branch(x)

        if self.use_identity:
            out += self.identity(x)
        return out

    def get_equivalent_kernel_bias(self):
        """将多分支融合为单一大核"""
        max_k = self.max_k
        pad_center = max_k // 2
        fused_k = torch.zeros(self.c1, 1, max_k, max_k, device=self.branches[0][0].weight.device)
        fused_b = torch.zeros(self.c1, device=self.branches[0][0].weight.device)

        # 融合卷积分支
        for branch in self.branches:
            conv, bn = branch[0], branch[1]
            k_val, b_val = self._fuse_bn(conv, bn)

            # 将小核填充到大核中心
            k_size = k_val.shape[-1]
            pad = (max_k - k_size) // 2
            fused_k += F.pad(k_val, [pad, pad, pad, pad])
            fused_b += b_val

        # 融合 Identity 分支
        if self.use_identity:
            k_id, b_id = self._fuse_bn_identity(self.identity)
            pad = (max_k - 1) // 2  # 1x1 到 7x7
            fused_k += F.pad(k_id, [pad, pad, pad, pad])
            fused_b += b_id

        return fused_k, fused_b

    def _fuse_bn(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_identity(self, bn):
        # Identity 相当于 1x1 卷积，权重为 1
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()

        k_val = torch.zeros(self.c1, 1, 1, 1, device=gamma.device)
        for i in range(self.c1):
            k_val[i, 0, 0, 0] = 1.0

        t = (gamma / std).reshape(-1, 1, 1, 1)
        return k_val * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.deploy: return
        k, b = self.get_equivalent_kernel_bias()
        self.dw_reparam = nn.Conv2d(self.c1, self.c1, self.max_k, self.stride, self.max_k // 2, groups=self.c1,
                                    bias=True)
        self.dw_reparam.weight.data = k
        self.dw_reparam.bias.data = b
        self.__delattr__('branches')
        if hasattr(self, 'identity'): self.__delattr__('identity')
        self.deploy = True


# ---------------------- 核心模块：StarBlock + LSK ----------------------
class StarLSK_Bottleneck(nn.Module):
    """
    基于 StarNet 与 Large Selective Kernel 思想设计的瓶颈模块。
    通过元素级乘法实现高阶特征交互，显著提升非线性表达能力。
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=7, e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        # 分支 1: 基础变换
        self.cv1 = Conv(c1, c_, 1, 1)
        # 分支 2: 大核感受野 (Large Kernel Depthwise)
        # 允许参数冗余：使用较大的卷积核捕捉长距离依赖
        self.dw = nn.Sequential(
            nn.Conv2d(c_, c_, k, 1, k // 2, groups=c_, bias=False),
            nn.BatchNorm2d(c_)
        )
        # 分支 3: 投影输出
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x1 = self.cv1(x)
        # Star Operation: x * Act(DW(x))
        # 这种乘法操作能模拟高维特征空间的交互
        y = x1 * F.silu(self.dw(x1))
        return x + self.cv2(y) if self.add else self.cv2(y)


# ---------------------- 封装给 YOLO 使用的 C3k2 变体 ----------------------
class C3k_StarLSK(C3):
    """
    C3k 的 Star 增强版。
    继承自 C3，内部使用 StarLSK_Bottleneck 替换原生 Bottleneck。
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道
        # 严格遵循 C3k 风格：n 个串联的 Bottleneck
        self.m = nn.Sequential(*(StarLSK_Bottleneck(c_, c_, shortcut, g, k=k, e=1.0) for _ in range(n)))
class C3k2_StarLSK(C2f):
    """
    C3k2 的 Star 增强版。
    继承自 C2f，支持切换 C3k_Star 或直接使用 StarLSK_Bottleneck。
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, k=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 根据 c3k 参数决定内部嵌套结构
        # self.c 是 C2f 父类计算好的隐藏通道 (c2 * e)
        self.m = nn.ModuleList(
            C3k_StarLSK(self.c, self.c, 2, shortcut, g, k=k) if c3k else
            StarLSK_Bottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )