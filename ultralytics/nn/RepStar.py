import torch
import torch.nn as nn
from timm.models.layers import DropPath
from ultralytics.nn.modules import Conv  # Ultralytics 框架的 Conv（conv bias=False + BN + optional act）
from ultralytics.nn.modules.block import C3k, C3k2


class RepDW(nn.Module):
    """
    Reparameterized Depthwise Conv（大核版本，对齐 Ultralytics RepConv 风格）。
    - 训练时：多分支 dwconv (7x7 + 5x5 + 3x3，每个带 BN) + identity BN
    - 推理时：融合成单一 7x7 dwconv（bias=True，无 BN，更快）
    """
    def __init__(self, c, kernels=[7, 5, 3]):
        super().__init__()
        self.c = c
        self.g = c  # depthwise
        self.kernels = kernels
        self.max_k = max(kernels)
        self.pad = self.max_k // 2

        # 训练分支：每个 kernel 一个 Conv（act=False，所以有 conv + BN）
        self.branches = nn.ModuleList([
            Conv(c, c, k, 1, k // 2, g=c, act=False) for k in kernels
        ])

        # identity BN
        self.id_bn = nn.BatchNorm2d(c)

        # 推理时融合后的 conv
        self.fused_conv = None

    def forward(self, x):
        if self.fused_conv is not None:
            return self.fused_conv(x)

        # 训练模式：多分支 + identity
        out = self.id_bn(x)
        for branch in self.branches:
            out = out + branch(x)
        return out

    def fuse(self):
        if self.fused_conv is not None:
            return

        # 先取 device（从现有参数取，避免空）
        device = next(self.id_bn.parameters()).device

        fused_kernel = torch.zeros((self.c, 1, self.max_k, self.max_k), device=device)
        fused_bias = torch.zeros(self.c, device=device)

        for branch in self.branches:
            k, b = self._fuse_conv_bn(branch)
            padded_k = self._pad_to_max(k, branch.conv.kernel_size[0])
            fused_kernel += padded_k
            fused_bias += b

        k_id, b_id = self._fuse_identity_bn()
        fused_kernel += k_id
        fused_bias += b_id

        # 创建融合后的 conv
        self.fused_conv = nn.Conv2d(
            in_channels=self.c,
            out_channels=self.c,
            kernel_size=self.max_k,
            stride=1,
            padding=self.pad,
            groups=self.g,
            bias=True
        )
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # 删除训练分支，释放内存
        for attr in ['branches', 'id_bn']:
            if hasattr(self, attr):
                delattr(self, attr)

    def _fuse_conv_bn(self, conv_module):
        """融合 Ultralytics Conv（conv + bn）"""
        conv = conv_module.conv
        bn = conv_module.bn
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_kernel = conv.weight * t
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        return fused_kernel, fused_bias

    def _fuse_identity_bn(self):
        """identity BN → center=1 的 kernel + bias"""
        bn = self.id_bn
        device = bn.weight.device
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)

        # center ysl center=1 的 identity kernel（shape: c,1,max_k,max_k）
        kernel = torch.zeros((self.c, 1, self.max_k, self.max_k), device=device)
        center = self.max_k // 2
        kernel[:, :, center, center] = 1.0
        fused_kernel = kernel * t
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        return fused_kernel, fused_bias

    def _pad_to_max(self, kernel, curr_k):
        """将小核 pad 到 max_k"""
        if curr_k == self.max_k:
            return kernel
        pad_total = self.max_k - curr_k
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        return nn.functional.pad(kernel, [pad_l, pad_r, pad_l, pad_r])

class RepStarBlock(nn.Module):
    """更新后的 RepStarBlock：使用 RepDW 替换第一个 depthwise 部分，对齐 Ultralytics 风格"""
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()

        # ---------- 第一个 depthwise：Reparameterized 大核 ----------
        self.rep_dw = RepDW(dim, kernels=[7, 5, 3])

        # ---------- Star channel mixing ----------
        # f1/f2：纯 1x1 conv（无 BN），使用 nn.Conv2d + bias=True
        expanded = int(dim * mlp_ratio)
        self.f1 = nn.Conv2d(dim, expanded, 1, bias=True)
        self.f2 = nn.Conv2d(dim, expanded, 1, bias=True)
        # g：1x1 压缩 + BN（使用 Ultralytics Conv，act=False）
        self.g = Conv(expanded, dim, 1, 1, act=False)
        self.act = nn.ReLU6()

        # ---------- 第二个 depthwise：纯 7x7 dwconv（无 BN） ----------
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        # 第一个 depthwise（训练多分支 / 推理单 7x7）
        x = self.rep_dw(x)

        # Star operation
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)

        # 第二个 depthwise
        x = self.dwconv2(x)

        # 残差
        x = input + self.drop_path(x)
        return x

    def switch_to_deploy(self):
        """切换到推理模式（融合 RepDW）"""
        self.rep_dw.fuse()





# 假设你已经定义了 RepStarBlock 和其中的 RepDW（如前几次对话）
# 如果需要 drop_path，可以后续扩展，这里先保持与原 MambaVision 集成方式一致（无 drop_path，或固定为 0）

class C3k_RepStar(C3k):
    """
    C3k 结构，但将中间的 Bottleneck / 普通块替换为堆叠的 RepStarBlock
    完全参照 C3k_MambaVision 的集成方式
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3, mlp_ratio=3.0):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepStarBlock(dim=c_, mlp_ratio=mlp_ratio, drop_path=0.) for _ in range(n)))


class C3k2_RepStar(C3k2):
    """
    C3k2 结构（通常是更灵活的堆叠版），参照 C3k2_MambaVision 的实现
    - 如果 c3k=True：每层用 C3k_RepStar（内部再堆 2 个 RepStarBlock）
    - 如果 c3k=False：每层直接用单个 RepStarBlock
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, mlp_ratio=3.0):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # self.c 是 Ultralytics C3k2 中通过 e 计算得到的 hidden dim（类似 c_）
        if c3k:
            self.m = nn.ModuleList(
                C3k_RepStar(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, mlp_ratio=mlp_ratio)
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                RepStarBlock(dim=self.c, mlp_ratio=mlp_ratio, drop_path=0.) for _ in range(n)
            )

    def switch_to_deploy(self):
        """
        推理部署模式：
        - 递归调用内部 RepStarBlock 的 switch_to_deploy
        - 仅做结构融合，不改 forward 行为
        """
        for m in self.m:
            # 情况 1：直接是 RepStarBlock
            if isinstance(m, RepStarBlock):
                m.switch_to_deploy()

            # 情况 2：是 C3k_RepStar（里面还包着 RepStarBlock）
            elif hasattr(m, "m"):
                for sub_m in m.m:
                    if isinstance(sub_m, RepStarBlock):
                        sub_m.switch_to_deploy()