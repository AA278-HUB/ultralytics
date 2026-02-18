# ==================== 必需的 helper 函数（UniRepLKNet 官方融合逻辑） ====================
import torch
import torch.nn.functional as F
from ultralytics import nn
from ultralytics.nn.modules import RepConv, C3, C2f, Conv


def fuse_bn(conv, bn):
    """Fuse BN into Conv"""
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std + conv_bias * t.squeeze()

def convert_dilated_to_nondilated(kernel, dilate_rate):
    """把 dilated kernel 转成等效非 dilated 大 kernel（稀疏填充）"""
    if kernel.size(1) != kernel.size(0):  # 非 DW 情况（我们这里是 DW）
        raise NotImplementedError("仅支持 DW")
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device, dtype=kernel.dtype)
    dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    return dilated

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    """把 dilated 分支 merge 到主大核里"""
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_k = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    pad = (large_k - equivalent_k) // 2
    if pad < 0:
        raise ValueError("分支等效核比主核大！")
    merged = large_kernel + F.pad(equivalent_kernel, [pad] * 4)
    return merged
# ==================== 最终加强版 LSKA（2025~2026 最流行大核注意力） ====================
class LSKA(nn.Module):
    """Large Separable Kernel Attention (LSKA) - 2025 年大量 YOLO 论文验证的 SOTA 空间注意力
    用 1xk + kx1 可分离大核，参数极低但感受野极大，比 CA 更适合检测任务"""
    def __init__(self, dim, k_size=21):   # 默认 21（超大感受野），可根据层数调小
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

# ==================== 可重参数化的大核模块（核心修复） ====================
class DilatedReparamConv(nn.Module):
    """UniRepLKNet 官方风格 Dilated Reparam（训练多分支，推理 fuse 成单个大核）"""
    def __init__(self, c, k=21, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.c = c

        # 主大核（目标感受野）
        pad = k // 2
        self.lk_origin = nn.Conv2d(c, c, k, stride=1, padding=pad, groups=c, bias=deploy)

        if not deploy:
            self.origin_bn = nn.BatchNorm2d(c)

            # 分支配置（根据 k 自动选择，平衡精度与参数）
            if k >= 29:
                self.kernel_sizes = [5, 7, 9, 11, 13, 5, 7, 9, 3, 3, 3, 3, 3, 3]
                self.dilates = [1, 1, 1, 1, 1, 2, 2, 2, 4, 6, 8, 10, 12, 14]
            elif k >= 21:
                self.kernel_sizes = [5, 7, 9, 11, 5, 7, 3, 3, 3, 3]
                self.dilates = [1, 1, 1, 1, 2, 2, 3, 5, 7, 9]
            elif k == 17:
                self.kernel_sizes = [5, 9, 3, 3, 3]
                self.dilates = [1, 2, 4, 5, 7]
            else:  # fallback
                self.kernel_sizes = [5, 7, 3, 3, 3]
                self.dilates = [1, 2, 3, 4, 5]

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
        """训练完后调用，融合成单个大核（参数/速度优化）"""
        if self.deploy:
            return

        # fuse 主大核
        origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)

        # fuse 所有分支并 merge
        for ks, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f'dil_conv_k{ks}_{r}')
            bn = getattr(self, f'dil_bn_k{ks}_{r}')
            branch_k, branch_b = fuse_bn(conv, bn)
            origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b += branch_b

        # 创建最终单 Conv（现在是纯大核）
        final_k = origin_k.shape[2]
        final_pad = final_k // 2
        self.lk_origin = nn.Conv2d(self.c, self.c, final_k, 1, final_pad, groups=self.c, bias=True)
        self.lk_origin.weight.data.copy_(origin_k)
        self.lk_origin.bias.data.copy_(origin_b)

        self.deploy = True

        # 删除训练分支
        for ks, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, f'dil_conv_k{ks}_{r}')
            delattr(self, f'dil_bn_k{ks}_{r}')
        delattr(self, 'origin_bn')


# ==================== v4.1 瓶颈 & C3k2（只需改这部分，其他保持不变） ====================
class EnhancedUniRepLK_Bottleneck_v4(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=21, k_attn=21, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k=3, s=1, g=g)
        self.large = DilatedReparamConv(c_, k=k, deploy=False)   # ← 现在是可融合版
        self.attn = LSKA(c_, k_size=k_attn)
        self.ffn = nn.Sequential(
            nn.Conv2d(c_, int(c_ * 1.5), 1, bias=False),
            nn.BatchNorm2d(int(c_ * 1.5)),
            nn.SiLU(),
            nn.Conv2d(int(c_ * 1.5), c_, 1, bias=False),
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
# ==================== v4 瓶颈（核心升级） ====================
class EnhancedUniRepLK_Bottleneck_v4(nn.Module):
    """v4 终极加强瓶颈：超强 dilated + LSKA + 高扩充 FFN"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=31, k_attn=21, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k=3, s=1, g=g)           # 局部
        self.large = DilatedReparamConv(c_, k=k)            # 超强全局多尺度
        self.attn = LSKA(c_, k_size=k_attn)                 # LSKA 代替 CA
        # FFN 扩充到 1.5x（更高非线性）
        self.ffn = nn.Sequential(
            nn.Conv2d(c_, int(c_ * 1.5), 1, bias=False),
            nn.BatchNorm2d(int(c_ * 1.5)),
            nn.SiLU(),
            nn.Conv2d(int(c_ * 1.5), c_, 1, bias=False),
            nn.BatchNorm2d(c_),
        )
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x
        x = self.cv1(x)
        x = self.large(x)
        x = self.attn(x)
        x = self.ffn(x) + x          # FFN 残差
        x = self.cv3(x)
        return residual + x if self.add else x


# ==================== v4 C3k 和 C3k2 ====================
class C3k_UniRepLK_Enhanced_v4(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, k=31, k_attn=21, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(EnhancedUniRepLK_Bottleneck_v4(c_, c_, shortcut, g, k=k, k_attn=k_attn, e=1.0)
              for _ in range(n))
        )


class C3k2_UniRepLKv4(C2f):
    """
    C3k2 的 v4 终极版（强烈推荐替换 v3）
    - c3k=True：更深（推荐精度最高）
    - k=31 / k_attn=21：最大感受野（参数多，精度最强）
    """
    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, g=1, shortcut=True, k=31, k_attn=21):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_UniRepLK_Enhanced_v4(self.c, self.c, 2, shortcut, g, k=k, k_attn=k_attn) if c3k else
            EnhancedUniRepLK_Bottleneck_v4(self.c, self.c, shortcut, g, k=k, k_attn=k_attn)
            for _ in range(n)
        )

# C3k_UniRepLK_Enhanced_v4 和 C3k2_UniRepLKv4 保持你上次的代码，只把里面的 EnhancedUniRepLK_Bottleneck_v4 改成上面的新版即可