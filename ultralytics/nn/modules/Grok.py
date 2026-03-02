# ==================== 必需的 helper 函数（UniRepLKNet 官方融合逻辑） ====================
import torch
import torch.nn.functional as F
from ultralytics.nn.modules import RepConv, C3, C2f, Conv
import torch.nn as nn

from ultralytics.nn.modules.conv import autopad


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
class ConvGELU(nn.Module):
    """自定义 Conv-BN-GELU 层，处理复杂特征。"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class LSKA(nn.Module):
    """Large-Scale Kernel Attention，受 RLRD-YOLO 启发。"""
    def __init__(self, channels: int, k_size: int = 7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )
        self.large_conv = nn.Conv2d(channels, channels, k_size, 1, k_size // 2, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)
        large = self.large_conv(x)
        return x * y + large  # 融合大核注意力

class MultiBranchAttn(nn.Module):
    """多分支 SENetV2-like 注意力，受 ADSPPF 启发。"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // reduction, 1), nn.ReLU(), nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid())
        self.branch2 = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Conv2d(channels, channels // reduction, 1), nn.ReLU(), nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return x * (b1 + b2) / 2  # 平均多分支

class C3TR(nn.Module):
    """Cross-Stage Partial Transformer，受 SC3T 启发。"""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.cv1 = ConvGELU(c1, c2 // 2, 1, 1)
        self.cv2 = ConvGELU(c1, c2 // 2, 1, 1)
        self.mha = nn.MultiheadAttention(c2 // 2, num_heads=4, batch_first=True)
        self.cv3 = ConvGELU(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        b, c, h, w = y2.shape
        y2 = y2.flatten(2).permute(0, 2, 1)
        y2, _ = self.mha(y2, y2, y2)
        y2 = y2.permute(0, 2, 1).view(b, c, h, w)
        return self.cv3(torch.cat([y1, y2], 1))

class LSKA_ASPPF(nn.Module):
    """重新设计的 SPPF，包含 LSKA、多尺度扩张卷积、多分支注意力、C3TR 和 shortcut。"""

    def __init__(self, c1: int, c2: int, rates: tuple = (1, 3, 5)):
        """
        初始化 LSKA_ASPPF 层。

        Args:
            c1 (int): 输入通道。
            c2 (int): 输出通道（通常 c2 == c1）。
            rates (tuple): 扩张率。

        Notes:
            扩展 SPP(k=(5,9,13))，添加 LSKA 以大核注意力、多尺度扩张卷积以细节捕捉、多分支注意力以融合、C3TR 以全局依赖，以及投影 shortcut 以匹配通道。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = ConvGELU(c1, c_, 1, 1)
        self.cv2 = ConvGELU(c_ * (len(rates) + 2), c2, 1, 1)  # +2 为 LSKA 和 C3TR
        self.aspp_convs = nn.ModuleList([nn.Conv2d(c_, c_, 3, 1, dilation=r, padding=r, bias=False) for r in rates])
        self.lska = LSKA(c_)
        self.mb_attn = MultiBranchAttn(c_)
        self.c3tr = C3TR(c_, c_)
        self.proj_mid = ConvGELU(c_, c2, 1, 1) if c_ != c2 else nn.Identity()  # 投影 mid 以匹配 c2
        self.proj_x = ConvGELU(c1, c2, 1, 1) if c1 != c2 else nn.Identity()  # 投影 x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用 LSKA、多尺度扩张卷积、多分支注意力、C3TR 并返回拼接特征图与投影 shortcut。"""
        y = [self.cv1(x)]  # 从 cv1 输出开始 (c_)
        mid = self.lska(y[-1])  # LSKA 后 mid (c_)
        for ac in self.aspp_convs:  # 多尺度扩张卷积
            aspp = ac(mid)
            attended = self.mb_attn(aspp)  # 多分支注意力
            y.append(attended)  # 每个 c_
        tr_out = self.c3tr(mid)  # C3TR (c_)
        y.append(tr_out)
        cat_features = torch.cat(y, 1)  # 拼接：c_ * (len(rates) + 2)
        output = self.cv2(cat_features)  # 到 c2
        return output + self.proj_mid(mid) + self.proj_x(x)  # 投影后多级 shortcut
# C3k_UniRepLK_Enhanced_v4 和 C3k2_UniRepLKv4 保持你上次的代码，只把里面的 EnhancedUniRepLK_Bottleneck_v4 改成上面的新版即可