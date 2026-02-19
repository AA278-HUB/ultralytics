import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f, Conv, C3, RepConv

# ==================== 融合辅助函数（必须有，用于 switch_to_deploy） ====================
def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std + conv_bias * t.squeeze()

def convert_dilated_to_nondilated(kernel, dilate_rate):
    if kernel.size(1) != kernel.size(0):
        raise NotImplementedError("仅支持 Depthwise")
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device, dtype=kernel.dtype)
    dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    return dilated

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_k = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    pad = (large_k - equivalent_k) // 2
    if pad < 0:
        raise ValueError("分支等效核比主核大！")
    merged = large_kernel + F.pad(equivalent_kernel, [pad] * 4)
    return merged

# ==================== LSKA（纳米友好，默认 k_size=7） ====================
class LSKA(nn.Module):
    def __init__(self, dim, k_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(1, k_size),
                               padding=(0, k_size//2), groups=dim, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(k_size, 1),
                               padding=(k_size//2, 0), groups=dim, bias=False)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        attn = self.conv1(x)
        attn = self.conv2(attn)
        attn = self.conv3(attn)
        return u * self.sigmoid(attn)

# ==================== DilatedReparamConv（纳米友好，默认 k=11，可 deploy） ====================
class DilatedReparamConv(nn.Module):
    def __init__(self, c, k=11, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.c = c

        pad = k // 2
        self.lk_origin = nn.Conv2d(c, c, k, stride=1, padding=pad, groups=c, bias=deploy)

        if not deploy:
            self.origin_bn = nn.BatchNorm2d(c)

            # 纳米友好分支配置（k=11 时自动选择）
            if k >= 13:
                self.kernel_sizes = [5, 7, 9, 5, 3, 3, 3]
                self.dilates = [1, 1, 1, 2, 3, 5, 7]
            else:
                self.kernel_sizes = [5, 7, 5, 3, 3]
                self.dilates = [1, 1, 2, 3, 5]

            for ks, r in zip(self.kernel_sizes, self.dilates):
                pad_b = r * (ks - 1) // 2
                setattr(self, f'dil_conv_k{ks}_{r}',
                        nn.Conv2d(c, c, ks, 1, pad_b, dilation=r, groups=c, bias=False))
                setattr(self, f'dil_bn_k{ks}_{r}', nn.BatchNorm2d(c))

    def forward(self, x):
        if self.deploy:
            return self.lk_origin(x)

        out = self.origin_bn(self.lk_origin(x))
        for ks, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f'dil_conv_k{ks}_{r}')
            bn = getattr(self, f'dil_bn_k{ks}_{r}')
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if self.deploy:
            return
        origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
        for ks, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f'dil_conv_k{ks}_{r}')
            bn = getattr(self, f'dil_bn_k{ks}_{r}')
            branch_k, branch_b = fuse_bn(conv, bn)
            origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b += branch_b

        final_k = origin_k.shape[2]
        final_pad = final_k // 2
        self.lk_origin = nn.Conv2d(self.c, self.c, final_k, 1, final_pad, groups=self.c, bias=True)
        self.lk_origin.weight.data.copy_(origin_k)
        self.lk_origin.bias.data.copy_(origin_b)

        self.deploy = True
        for ks, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, f'dil_conv_k{ks}_{r}')
            delattr(self, f'dil_bn_k{ks}_{r}')
        delattr(self, 'origin_bn')

# ==================== v5 瓶颈（FFN 1.2x，纳米友好） ====================
class EnhancedUniRepLK_Bottleneck_v5(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=11, k_attn=7, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k=3, s=1, g=g)          # 你原来的 RepConv
        self.large = DilatedReparamConv(c_, k=k, deploy=False)
        self.attn = LSKA(c_, k_size=k_attn)
        self.ffn = nn.Sequential(
            nn.Conv2d(c_, int(c_ * 1.2), 1, bias=False),
            nn.BatchNorm2d(int(c_ * 1.2)),
            nn.SiLU(),
            nn.Conv2d(int(c_ * 1.2), c_, 1, bias=False),
            nn.BatchNorm2d(c_),
        )
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x
        x = self.cv1(x)
        x = self.large(x)
        x = self.attn(x)
        x = self.ffn(x) + x
        x = self.cv3(x)
        return residual + x if self.add else x

# ==================== C3k v5 ====================
class C3k_UniRepLK_Enhanced_v5(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, k=11, k_attn=7, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(EnhancedUniRepLK_Bottleneck_v5(c_, c_, shortcut, g, k=k, k_attn=k_attn, e=1.0)
              for _ in range(n))
        )

# ==================== C3k2_UniRepLKv5（最终类） ====================
class C3k2_UniRepLKv5(C2f):
    """
    YOLO11n 专用 v5 版（默认 k=11, k_attn=7）
    使用方法：
    C3k2_UniRepLKv5(c2, c3k=True/False, e=0.5, k=11, k_attn=7)
    """
    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, g=1, shortcut=True, k=11, k_attn=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_UniRepLK_Enhanced_v5(self.c, self.c, 2, shortcut, g, k=k, k_attn=k_attn) if c3k else
            EnhancedUniRepLK_Bottleneck_v5(self.c, self.c, shortcut, g, k=k, k_attn=k_attn)
            for _ in range(n)
        )