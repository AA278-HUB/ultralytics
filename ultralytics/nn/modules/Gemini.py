import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2f, Conv
from ultralytics.nn.modules.block import C3k
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


class Dense_GLK_Block(nn.Module):
    """
    Dense Gated Large-Kernel Block (Dense-GLK)

    设计理念 (Design Principles):
    1. Dense-Heavy: 使用标准卷积替代 1x1 点卷积，参数量大幅提升，换取极强的信息混合能力。
    2. Large-Kernel Context: 中间层嵌入 7x7 (或更大) 深度卷积，提供宽阔感受野。
    3. Channel Expansion: 在 Block 内部将通道数扩展 2 倍 (Wide Structure)，捕获更多高维特征。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=7, e=0.5):
        """
        Args:
            c1: 输入通道
            c2: 输出通道
            k: 大核尺寸 (建议 7 或 9)
        """
        super().__init__()
        c_ = int(c2 * e)  # C2f 传入的隐藏通道数

        # 内部扩展因子，设为 2.0 以大幅增加参数量和特征维度
        self.inner_ratio = 2.0
        c_inner = int(c_ * self.inner_ratio)

        # 1. 前置密集卷积 (Heavy Dense Conv): 3x3, c -> 2c
        # 这一步提供了主要的参数量来源 (9 * c * 2c)
        self.cv1 = Conv(c1, c_inner, k=3, s=1)

        # 2. 大核深度卷积 (Large Kernel DW): 7x7
        # 捕捉长距离依赖，虽然参数少，但对精度至关重要
        self.lk_dw = nn.Conv2d(c_inner, c_inner, k, 1, autopad(k), groups=c_inner, bias=False)
        self.bn_lk = nn.BatchNorm2d(c_inner)

        # 3. 门控分支 (Gating Branch)
        # 简单的注意力机制：原特征 * 经过大核处理的特征
        # 这里没有额外的 sigmoid，利用 SiLU 的特性进行自门控

        # 4. 后置压缩卷积 (Heavy Projection): 3x3, 2c -> c
        # 再次使用 3x3 而不是 1x1，进一步增加参数量和空间融合能力
        self.cv2 = Conv(c_inner, c2, k=3, s=1)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 1. 升维与初步特征提取
        y = self.cv1(x)

        # 2. 大核上下文提取 (DWConv)
        context = self.bn_lk(self.lk_dw(y))

        # 3. 门控融合 (Star Operation / Gating)
        # y 是高频细节，context 是低频语义上下文，两者相乘
        y = y * context

        # 4. 降维与输出
        y = self.cv2(y)

        return x + y if self.add else y


class C3k_DenseGLK(C3k):
    """继承 C3k，将内部 Bottleneck 替换为 Heavy 的 Dense_GLK_Block。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # 使用 Dense_GLK_Block，内部不再需要 e=1.0 的限制，因为 Block 内部自己做了 expand
        self.m = nn.Sequential(*(Dense_GLK_Block(c_, c_, shortcut, g, k=k) for _ in range(n)))


class C3k2_DenseGLK(C2f):
    """
    C3k2 的 Heavy 版本。
    使用说明：在 yaml 文件中将 C2f 或 C3k2 替换为 C3k2_DenseGLK。
    """

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, k=7, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 无论 c3k 为 True 还是 False，我们都强制使用增强版模块
        # 这里的 e=1.0 是传给 Block 的，因为 C2f 已经做了一次 split，
        # 我们希望 Block 内部处理全部传入的 hidden channels
        self.m = nn.ModuleList(
            C3k_DenseGLK(self.c, self.c, 2, shortcut, g, e=1.0, k=k) if c3k else
            Dense_GLK_Block(self.c, self.c, shortcut, g, k=k) for _ in range(n)
        )


class GRN(nn.Module):
    """
    Global Response Normalization (GRN) layer from ConvNeXt V2.
    Enhances feature competition and contrast.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W) -> permute to (B, H, W, C) for layer norm style logic
        x = x.permute(0, 2, 3, 1)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        x = self.gamma * (x * Nx) + self.beta + x
        x = x.permute(0, 3, 1, 2)
        return x
class StarRepLK_Block(nn.Module):
    """
    StarRepLK Block:
    1. RepConv for Local Features (3x3).
    2. Large Kernel Depthwise Conv for Global Context (7x7).
    3. Star Operation (Element-wise Multiplication) for high-order non-linearity.
    4. GRN for feature contrast.
    5. SEBlock for channel attention.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=7, e=0.5):
        super().__init__()
        self.c = c2  # Output channels

        # 1. Expansion & Local Feature Extraction (RepConv)
        # We expand to 2x channels to facilitate the "Star" split later
        self.local_rep = RepConv(c1, 2 * c2, k=3, s=1, g=g)

        # 2. Global Context (Large Kernel Depthwise)
        # Applied on the expanded features to capture context before gating
        padding = k // 2
        self.global_dw = nn.Conv2d(2 * c2, 2 * c2, k, stride=1, padding=padding, groups=2 * c2, bias=False)
        self.bn_dw = nn.BatchNorm2d(2 * c2)

        # 3. Feature Processing components
        self.grn = GRN(c2)
        self.attn = SEBlock(c2, ratio=16)  # Channel Attention

        # 4. Final Projection
        self.proj = nn.Conv2d(c2, c2, 1, 1, bias=False)
        self.bn_proj = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()  # Simple activation before projection if needed
        self.add = shortcut and c1 == c2

    def forward(self, x):
        input_tensor = x

        # Step 1: Strong Local Features
        x = self.local_rep(x)

        # Step 2: Strong Global Context
        x = self.global_dw(x)
        x = self.bn_dw(x)

        # Step 3: Star Operation (Gating)
        # Split channels into two halves
        x1, x2 = x.chunk(2, dim=1)
        # Element-wise multiplication: This is the "Star" operation
        # One branch gates the other.
        x = x1 * x2

        # Step 4: Refinement
        x = self.grn(x)  # Global Response Normalization
        x = self.attn(x)  # Channel Attention (SE)

        # Step 5: Projection
        x = self.proj(x)
        x = self.bn_proj(x)

        return input_tensor + x if self.add else x


class C3k2_StarRepLK(C2f):
    """
    Upgraded CSP Bottleneck with StarRepLK Blocks.
    Focus: High Accuracy through Gating, Reparameterization, and Global Context.
    """

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, k=7, g=1, shortcut=True):
        """
        Args:
            c1, c2: Input/Output channels
            n: Number of blocks
            e: Expansion ratio for the hidden channels in C2f wrapper
            k: Kernel size for the Large Kernel part (default 7)
        """
        super().__init__(c1, c2, n, shortcut, g, e)

        # Replace the standard bottleneck list with our StarRepLK_Block
        # self.c is the hidden channel size calculated in C2f
        self.m = nn.ModuleList(
            StarRepLK_Block(self.c, self.c, shortcut, g, k=k) for _ in range(n)
        )