import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2f


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


class UniRepLK_Block(nn.Module):
    """
    Universal Reparam Large Kernel Block.
    结构: RepConv(3x3) -> RepDWConv(7x7/9x9) -> SE -> Conv(1x1)
    参数量: 相比普通Bottleneck显著增加 (主要来自RepConv的稠密计算)
    精度: 结合了局部高频特征(3x3)和全局感受野(7x7)以及通道注意力。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=7, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        # 1. 局部特征提取 & 升维/降维
        # 这里使用 RepConv (3x3 Dense) 替代了原本的 1x1 Conv
        # 3x3 卷积的参数量是 1x1 的 9 倍，这将大幅提高模型的容量和拟合能力
        self.loc_feat = RepConv(c1, c_, k=3, s=1, g=1)  # 这里的 g=1 保证是 Dense 的，吃参数大户

        # 2. 全局特征建模 (Large Kernel)
        # 深度可分离卷积，使用大核 (k=7 或 9)
        # 这里为了保持简洁，使用标准 Conv+BN，如果显存允许，也可以换成 RepDW
        padding = k // 2
        self.glob_feat = nn.Sequential(
            nn.Conv2d(c_, c_, k, 1, padding, groups=c_, bias=False),  # Depthwise
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # 3. 通道注意力
        self.attn = SEBlock(c_, ratio=8)  # ratio越小参数越多，精度通常越高

        # 4. 投影输出
        self.proj = nn.Sequential(
            nn.Conv2d(c_, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x_loc = self.loc_feat(x)  # 强力的局部特征提取
        x_glob = self.glob_feat(x_loc)  # 大感受野
        x_attn = self.attn(x_glob)  # 关键特征筛选
        y = self.proj(x_attn)  # 融合
        return x + y if self.add else y


class C3k2_UniRepLK(C2f):
    """
    基于 UniRepLK_Block 的 C3k2 模块。
    """

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, k=7, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 注意：这里 e 是 hidden channel 的比例。
        # 由于我们使用了 RepConv(3x3)，参数量本身已经很大。
        # 如果依然觉得参数增加不够，可以将 e 调整为 0.6 或 0.7
        self.m = nn.ModuleList(
            UniRepLK_Block(self.c, self.c, shortcut, g, k=k, e=1.0) for _ in range(n)
        )