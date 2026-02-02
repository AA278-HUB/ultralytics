import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from ultralytics.nn.modules import Conv


class RepDW_Optimized(nn.Module):
    """
    优化后的 RepDW：
    1. 移除 5x5 分支，保留 7x7(全局) + 3x3(局部) + Identity(透传)。
    2. 这种组合在 UniRepLKNet 中被证明性价比最高。
    """

    def __init__(self, c, k=7):
        super().__init__()
        self.c = c
        self.max_k = k
        self.pad = k // 2

        # 训练时分支：
        # 1. 大核 7x7
        self.branch_large = Conv(c, c, k, 1, p=self.pad, g=c, act=False)
        # 2. 小核 3x3 (捕捉细节)
        self.branch_small = Conv(c, c, 3, 1, p=1, g=c, act=False)
        # 3. Identity BN
        self.branch_id = nn.BatchNorm2d(c)

        # 推理时的融合卷积
        self.fused_conv = None

    def forward(self, x):
        if self.fused_conv is not None:
            return self.fused_conv(x)

        # 训练模式：多分支相加
        out = self.branch_large(x) + self.branch_small(x) + self.branch_id(x)
        return out

    def fuse(self):
        if self.fused_conv is not None:
            return

        device = next(self.branch_large.parameters()).device

        # 1. 初始化融合核 (c, 1, k, k)
        fused_k = torch.zeros((self.c, 1, self.max_k, self.max_k), device=device)
        fused_b = torch.zeros(self.c, device=device)

        # 2. 融合 Large Kernel (7x7)
        k_l, b_l = self._fuse_conv_bn(self.branch_large)
        fused_k += k_l
        fused_b += b_l

        # 3. 融合 Small Kernel (3x3 -> Pad to 7x7)
        k_s, b_s = self._fuse_conv_bn(self.branch_small)
        fused_k += self._pad_center(k_s, self.max_k)
        fused_b += b_s

        # 4. 融合 Identity BN
        k_id, b_id = self._fuse_id(self.branch_id)
        fused_k += k_id
        fused_b += b_id

        # 5. 创建部署卷积
        self.fused_conv = nn.Conv2d(
            self.c, self.c, self.max_k, 1, self.pad, groups=self.c, bias=True
        )
        self.fused_conv.weight.data = fused_k
        self.fused_conv.bias.data = fused_b

        # 清理显存
        del self.branch_large, self.branch_small, self.branch_id

    def _fuse_conv_bn(self, branch):
        conv = branch.conv
        bn = branch.bn
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _fuse_id(self, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        # 创建中心为 1 的卷积核
        k = torch.zeros((self.c, 1, self.max_k, self.max_k), device=bn.weight.device)
        center = self.max_k // 2
        k[:, :, center, center] = 1.0
        return k * t, bn.bias - bn.running_mean * bn.weight / std

    def _pad_center(self, kernel, target_k):
        # 将小核 pad 到大核中心
        current_k = kernel.shape[-1]
        pad = (target_k - current_k) // 2
        return F.pad(kernel, [pad, pad, pad, pad])


class UniRepStarBlock(nn.Module):
    """
    改进版 StarBlock：
    Standard: DW(Rep) -> Star Mixing -> Add (with LayerScale)
    """

    def __init__(self, dim, mlp_ratio=4, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        # 1. 空间混合：RepDW (7x7 + 3x3)
        self.dw = RepDW_Optimized(dim, k=7)

        # 2. Star Operation Channel Mixing
        # 使用 mlp_ratio 扩展通道
        hidden_dim = int(dim * mlp_ratio)

        # FC1 & FC2 (用 Conv2d 1x1 实现)
        self.f1 = nn.Conv2d(dim, hidden_dim, 1)
        self.f2 = nn.Conv2d(dim, hidden_dim, 1)

        # Star Activation (SiLU is better for YOLO/Detection)
        self.act = nn.SiLU()

        # Output Projection
        self.g = nn.Conv2d(hidden_dim, dim, 1)

        # 3. LayerScale (关键！让网络更容易训练深层)
        self.layer_scale = nn.Parameter(
            torch.ones(dim, 1, 1) * layer_scale_init_value, requires_grad=True
        ) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_tensor = x

        # Step 1: Spatial Mixing (Reparameterized)
        x = self.dw(x)

        # Step 2: Star Operation (High dim interaction)
        # (X @ W1) * Act(X @ W2) -> StarNet/LLaMA Gated MLP style
        x1 = self.f1(x)
        x2 = self.f2(x)
        x = self.act(x1) * x2  # Element-wise multiplication

        # Step 3: Projection back
        x = self.g(x)

        # Step 4: LayerScale & Residual
        if self.layer_scale is not None:
            x = x * self.layer_scale

        x = input_tensor + self.drop_path(x)
        return x

    def switch_to_deploy(self):
        self.dw.fuse()