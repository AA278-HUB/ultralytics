import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f, Conv, C3, RepConv
from ultralytics.nn.modules.conv import autopad


# ==================== 融合辅助函数（必须有，用于 switch_to_deploy） ====================
def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std + conv_bias * t.squeeze()

# def convert_dilated_to_nondilated(kernel, dilate_rate):
#     if kernel.size(1) != kernel.size(0):
#         raise NotImplementedError("仅支持 Depthwise")
#     identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device, dtype=kernel.dtype)
#     dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
#     return dilated
#
# def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
#     large_k = large_kernel.size(2)
#     dilated_k = dilated_kernel.size(2)
#     equivalent_k = dilated_r * (dilated_k - 1) + 1
#     equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
#     pad = (large_k - equivalent_k) // 2
#     if pad < 0:
#         raise ValueError("分支等效核比主核大！")
#     merged = large_kernel + F.pad(equivalent_kernel, [pad] * 4)
#     return merged
import torch.nn.functional as F


def convert_dilated_to_nondilated(kernel, dilate_rate):
    """
    將空洞卷積核轉換為等效的普通卷積核 (填充零)。
    """
    # 修正：Depthwise 卷積的 weight 形狀是 [C, 1, K, K]
    # 我們要確保第二維是 1，這代表它是深度可分離卷積
    if kernel.size(1) != 1:
        raise NotImplementedError(f"僅支持 Depthwise 卷積，當前輸入維度為 {kernel.size(1)}")

    # 創建一個 1x1 的單位矩陣作為 Transpose Conv 的權重，用於插值填充零
    # 這裡要匹配卷積核的設備和數據類型
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device, dtype=kernel.dtype)

    # 通過 conv_transpose2d 在卷積核元素之間插入 (dilate_rate - 1) 個零
    dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate, groups=kernel.size(0))
    return dilated


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    """
    將擴展後的空洞卷積權重累加到主權重中。
    """
    large_k = large_kernel.size(2)

    # 1. 轉換空洞核為等效密集核
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    equivalent_k = equivalent_kernel.size(2)

    # 2. 計算需要補齊的 Padding
    pad = (large_k - equivalent_k) // 2
    if pad < 0:
        # 如果空洞核擴張後比主核還大，需要對主核進行 Padding (通常不會發生，除非參數設置錯誤)
        print(f"Warning: equivalent_k ({equivalent_k}) > large_k ({large_k})")
        return equivalent_kernel + F.pad(large_kernel, [abs(pad)] * 4)

    # 3. 將等效核對齊並相加
    return large_kernel + F.pad(equivalent_kernel, [pad] * 4)

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
        self.k = k
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
    # ==================== 融合逻辑开始 ====================
    def _fuse_bn_tensor(self, conv, bn):
        """ 将 Conv 和 BN 的参数融合为等效的权重和偏置 """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        # 融合后的权重: W_fused = W * (gamma / std)
        fused_weight = kernel * t
        # 融合后的偏置: B_fused = beta - mean * (gamma / std)
        fused_bias = beta - running_mean * gamma / std
        return fused_weight, fused_bias

    def _dilate_tensor(self, weight, dilation):
        """ 将带有 dilation 的卷积核膨胀为等效的密集卷积核 """
        if dilation == 1:
            return weight

        # weight shape: [C, 1, K, K] (depthwise)
        C, _, K, _ = weight.shape
        new_K = (K - 1) * dilation + 1
        # 创建一个全 0 的大核
        dilated_weight = torch.zeros((C, 1, new_K, new_K), device=weight.device)
        # 将原始权重填入对应的空洞位置
        dilated_weight[:, :, ::dilation, ::dilation] = weight
        return dilated_weight

    def get_equivalent_kernel_bias(self):
        """ 获取所有分支融合后的最终卷积核与偏置 """
        # 1. 融合主分支
        fused_k, fused_b = self._fuse_bn_tensor(self.lk_origin, self.origin_bn)

        # 2. 遍历并融合所有空洞卷积分支
        for ks, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f'dil_conv_k{ks}_{r}')
            bn = getattr(self, f'dil_bn_k{ks}_{r}')

            # 先融合 BN
            k_branch, b_branch = self._fuse_bn_tensor(conv, bn)
            # 再处理 dilation，将其膨胀成密集矩阵
            k_branch = self._dilate_tensor(k_branch, r)

            # 3. 将结果加到主分支上
            # 需要先对齐卷积核大小 (pad 到 self.k)
            pad_size = (self.k - k_branch.shape[2]) // 2
            if pad_size > 0:
                k_branch = F.pad(k_branch, [pad_size, pad_size, pad_size, pad_size])

            fused_k += k_branch
            fused_b += b_branch

        return fused_k, fused_b

    def switch_to_deploy(self):
        """ 切换到推理模式：执行融合并删除多余分支 """
        if self.deploy:
            return

        fused_k, fused_b = self.get_equivalent_kernel_bias()

        # 重新初始化一个带有 bias 的卷积层
        self.lk_origin = nn.Conv2d(self.c, self.c, self.k, stride=1,
                                   padding=self.k // 2, groups=self.c, bias=True)
        self.lk_origin.weight.data = fused_k
        self.lk_origin.bias.data = fused_b

        # 删除训练分支
        for ks, r in zip(self.kernel_sizes, self.dilates):
            self.__delattr__(f'dil_conv_k{ks}_{r}')
            self.__delattr__(f'dil_bn_k{ks}_{r}')
        if hasattr(self, 'origin_bn'):
            self.__delattr__('origin_bn')

        self.deploy = True
        print(f"Successfully fused DilatedReparamConv (k={self.k})")

    # def switch_to_deploy(self):
    #     if self.deploy:
    #         return
    #     origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
    #     for ks, r in zip(self.kernel_sizes, self.dilates):
    #         conv = getattr(self, f'dil_conv_k{ks}_{r}')
    #         bn = getattr(self, f'dil_bn_k{ks}_{r}')
    #         branch_k, branch_b = fuse_bn(conv, bn)
    #         origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
    #         origin_b += branch_b
    #
    #     final_k = origin_k.shape[2]
    #     final_pad = final_k // 2
    #     self.lk_origin = nn.Conv2d(self.c, self.c, final_k, 1, final_pad, groups=self.c, bias=True)
    #     self.lk_origin.weight.data.copy_(origin_k)
    #     self.lk_origin.bias.data.copy_(origin_b)
    #
    #     self.deploy = True
    #     for ks, r in zip(self.kernel_sizes, self.dilates):
    #         delattr(self, f'dil_conv_k{ks}_{r}')
    #         delattr(self, f'dil_bn_k{ks}_{r}')
    #     delattr(self, 'origin_bn')

# ==================== v5 瓶颈（FFN 1.2x，纳米友好） ====================
class EnhancedUniRepLK_Bottleneck_v5(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=11, k_attn=7, e=0.5):
        super().__init__()
        c_ = int(c2 * e*1)
        self.cv1 = RepConv(c1, c_, k=3, s=1, g=g)          # 你原来的 RepConv
        self.large = DilatedReparamConv(c_, k=k, deploy=False)
        self.attn = LSKA(c_, k_size=k_attn)
        self.ffn = nn.Sequential(
            nn.Conv2d(c_, int(c_ * 2), 1, bias=False),
            nn.BatchNorm2d(int(c_ * 2)),
            nn.SiLU(),
            nn.Conv2d(int(c_ *2), c_, 1, bias=False),
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

    def fuse(self):
        """
        深度融合：遍历所有子层级，融合 RepConv 和 DilatedReparamConv。
        """
        print(f"Fusing {self.__class__.__name__} stages...")
        # 使用 self.modules() 可以递归遍历所有子模块，不论嵌套多深
        for m in self.modules():
            # 1. 融合你定义的 RepConv (3x3 密集卷积)
            if isinstance(m, RepConv) and hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()

            # 2. 融合你定义的 DilatedReparamConv (大核空洞卷积)
            if isinstance(m, DilatedReparamConv) and hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()

        print(f"{self.__class__.__name__} fusion complete.")



class EMA_Attention(nn.Module):
    """
    Efficient Multi-Scale Attention (2024 常用增强模块)
    相比 SEBlock，它能更好地保留空间位置信息。
    """

    def __init__(self, channels, factor=8):
        super(EMA_Attention, self).__init__()
        self.groups = factor
        self.group_channels = channels // factor
        self.conv1x1 = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channels // factor, channels // factor)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_reshaped = x.view(b * self.groups, self.group_channels, h, w)
        x_avg = self.pool(x_reshaped)
        x1 = self.conv1x1(x_avg)
        x2 = self.conv3x3(x_avg)
        out = self.sigmoid(x1 + x2)
        return (x_reshaped * out).view(b, c, h, w)


class MultiScaleRepLK(nn.Module):
    """
    多尺度大核重参数化模块
    利用参数冗余：并行执行 3x3, 7x7, 11x11(DW) 提取多尺度空间特征
    """

    def __init__(self, dim, k=7):
        super().__init__()
        self.dim = dim
        # 局部密集路径
        self.rep_conv = RepConv(dim, dim, k=3)

        # 多尺度深度卷积路径 (捕捉不同感受野)
        self.dw_7x7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.dw_k = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # 融合局部与中远程特征
        return self.act(self.bn(self.rep_conv(x) + self.dw_7x7(x) + self.dw_k(x)))


class Dynamic_RSB_Block(nn.Module):
    """
    重磅设计的最终单元：高冗余、高精度版
    结构：Rep-Expansion -> Multi-Scale LK -> EMA Attention -> Project
    """

    def __init__(self, c1, c2, k=7, shortcut=True, e=1.0):  # e=1.0 极大增加冗余参数
        super().__init__()
        c_ = int(c2 * e)

        # 1. 强力升维层：使用 RepConv 替代 普通 1x1
        self.expansion = RepConv(c1, c_, k=3)

        # 2. 多尺度大核建模
        self.ms_lk = MultiScaleRepLK(c_, k=k)

        # 3. 增强型注意力
        self.attn = EMA_Attention(c_)

        # 4. 投影输出
        self.proj = nn.Sequential(
            nn.Conv2d(c_, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        res = x
        x = self.expansion(x)
        x = self.ms_lk(x)
        x = self.attn(x)
        x = self.proj(x)
        return x + res if self.add else x


class C3k2_Dynamic_RSB(nn.Module):
    """
    适配 YOLOv11/v10/v8 的 C3k2 封装
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=7):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 使用更加密集的模块堆叠
        self.m = nn.ModuleList(
            Dynamic_RSB_Block(self.c, self.c, k=k, shortcut=shortcut, e=1.5)  # e=1.5 内部通道扩增
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ConvReLU(nn.Module):
    """自定义 Conv-BN-ReLU 层，替换 SiLU 以提高效率。"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class HybridSPPF(nn.Module):
    """重新设计的 Spatial Pyramid Pooling - Fast，包含混合池化、ReLU 和 shortcut。"""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        初始化 HybridSPPF 层。

        Args:
            c1 (int): 输入通道。
            c2 (int): 输出通道（通常 c2 == c1）。
            k (int): 池化核大小。

        Notes:
            等价于 SPP(k=(5, 9, 13))，但使用混合（max + avg）池化以获得更好特征、
            ReLU 以加速计算，以及残差 shortcut 以改善训练。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = ConvReLU(c1, c_, 1, 1)
        self.cv2 = ConvReLU(c_ * 4, c2, 1, 1)
        self.m_max = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m_avg = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用混合池化操作，返回拼接的特征图并添加 shortcut。"""
        y = [self.cv1(x)]  # 从 cv1 输出开始
        for _ in range(3):  # 三级链式混合池化
            last = y[-1]
            max_p = self.m_max(last)
            avg_p = self.m_avg(last)
            hybrid = max_p + avg_p  # 混合：添加 max 和 avg 以获得鲁棒特征
            y.append(hybrid)
        cat_features = torch.cat(y, 1)  # 拼接：形状兼容 (c_ * 4)
        output = self.cv2(cat_features)
        return output + x  # 残差 shortcut（假设 c2 == c1）
class SEAttention(nn.Module):
    """简单通道注意力模块（SE Block）。"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentiveSPPF(nn.Module):
    """重新设计的 SPPF，包含注意力、多核池化和 shortcut。"""

    def __init__(self, c1: int, c2: int, kernels: tuple = (3, 5, 7)):
        """
        初始化 AttentiveSPPF 层。

        Args:
            c1 (int): 输入通道。
            c2 (int): 输出通道（通常 c2 == c1）。
            kernels (tuple): 多核池化大小。

        Notes:
            扩展 SPP(k=(3,5,7))，添加通道注意力以突出特征、多核以丰富细节，以及 shortcut 以改善流动。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = ConvReLU(c1, c_, 1, 1)
        self.cv2 = ConvReLU(c_ * (len(kernels) + 1), c2, 1, 1)  # +1 为 cv1 输出
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernels])
        self.attn = SEAttention(c_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用多核池化、注意力并返回拼接特征图与 shortcut。"""
        y = [self.cv1(x)]  # 从 cv1 输出开始
        for pool in self.pools:  # 多核链式池化
            pooled = pool(y[-1])
            attended = self.attn(pooled)  # 应用注意力
            y.append(attended)
        cat_features = torch.cat(y, 1)  # 拼接：形状兼容 (c_ * (len(kernels) + 1))
        output = self.cv2(cat_features)
        return output + x  # 残差 shortcut（假设 c2 == c1）

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
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

class CoordSPPF(nn.Module):
    """Modified SPPF with Coordinate Attention for better localization."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.att = CoordAtt(c_ * 4, c_ * 4) # 在拼接后进行注意力加权

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # 拼接特征
        combined = torch.cat((x, y1, y2, y3), 1)
        # 注入空间坐标注意力
        out = self.att(combined)
        return self.cv2(out)


import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """辅助函数：卷积 + BN"""
    result = nn.Sequential()
    result.add_module('conv',
                      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(out_channels))
    return result


class RepBranch(nn.Module):
    """训练时多分支，推理时单分支的重参数化模块"""

    def __init__(self, c, k=3, deploy=False):
        super().__init__()
        self.deploy = deploy
        if deploy:
            self.rbr_reparam = nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=True)
        else:
            self.rbr_dense = conv_bn(c, c, k, 1, k // 2, groups=c)
            self.rbr_1x1 = conv_bn(c, c, 1, 1, 0, groups=c)
            self.rbr_identity = nn.BatchNorm2d(c) if k == 3 else None

    def forward(self, x):
        if self.deploy:
            return self.rbr_reparam(x)

        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity:
            out += self.rbr_identity(x)
        return out


class RepOmniSPPF(nn.Module):
    """
    基于重参数化设计的高精度 SPPF
    目标：利用训练时的冗余分支提升检测精度 (mAP)，推理时合并。
    """

    def __init__(self, c1, c2, k=5, deploy=False):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, 1, 1), nn.BatchNorm2d(c_), nn.SiLU())
        self.cv2 = nn.Sequential(nn.Conv2d(c_ * 4, c2, 1, 1), nn.BatchNorm2d(c2), nn.SiLU())

        # 核心改进：将普通的 MaxPool 包装在重参数化增强块中
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 训练时为池化层增加并行的学习路径
        self.rep_blocks = nn.ModuleList([
            RepBranch(c_, k=3, deploy=deploy) for _ in range(3)
        ])

        # 最终特征融合后的超感受野增强
        self.rep_fusion = RepBranch(c_ * 4, k=7, deploy=deploy)

    def forward(self, x):
        x = self.cv1(x)

        # 串行池化结合重参数化分支
        y = [x]
        for i in range(3):
            # 既有池化的纹理保留，又有卷积的学习能力
            pool_out = self.m(y[-1])
            rep_out = self.rep_blocks[i](y[-1])
            y.append(pool_out + rep_out)  # 融合学习

        # 拼接后的全局特征增强
        z = torch.cat(y, 1)
        z = self.rep_fusion(z)

        return self.cv2(z)\

class GCBlock(nn.Module):
    """Global Context Block: 捕获全局长程依赖，增强对比度"""

    def __init__(self, in_channels, ratio=0.25):
        super(GCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = int(in_channels * ratio)
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # Context Modeling
        input_x = input_x.view(batch, channel, height * width)  # [B, C, N]
        mask = self.conv_mask(x).view(batch, 1, height * width)  # [B, 1, N]
        mask = self.softmax(mask)
        context = torch.matmul(input_x, mask.transpose(1, 2))  # [B, C, 1]
        context = context.view(batch, channel, 1, 1)
        # Transform & Fusion
        out = self.channel_add_conv(context)
        return x + out


class HCD_SPPF(nn.Module):
    """
    Hyper-Contextual Dynamic SPPF (HCD-SPPF)
    终极精度版：结合大核重参、全局上下文与跨维度注意力
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, 1, 1), nn.BatchNorm2d(c_), nn.SiLU())

        # 1. 传统多尺度池化路径 (保留基础纹理)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 2. 深度大核重参数化路径 (Rep-Large-Kernel)
        # 训练时使用大核增强感受野，推理时可融合
        self.large_kernel = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=7, padding=3, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.Conv2d(c_, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # 3. 全局上下文路径
        self.gc_block = GCBlock(c_)

        # 4. 融合后的特征精炼：使用 EMA 或 SimAM 增强
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 空间注意力加持
        self.spatial_att = nn.Sequential(
            nn.Conv2d(c_ * 4, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cv1(x)

        # 多支路特征并行提取
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.large_kernel(x)  # 大核特征
        y4 = self.gc_block(x)  # 全局上下文特征

        # 特征拼接 [x, y1, y2, y3, y4] 这里我们选择混合四支路
        feat = torch.cat([y1, y2, y3, y4], 1)

        # 空间权重精炼 (根据重要性重新分配特征权重)
        att = self.spatial_att(feat)
        feat = feat * att

        return self.cv2(feat)


class EA_SPPF(nn.Module):
    """
    Enhanced Attention Spatial Pyramid Pooling - Fast (EA-SPPF)
    改进点：引入混合池化、坐标注意力(CA)以及残差结构。
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        # 使用相同stride的池化层
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 引入坐标注意力机制 (Coordinate Attention)
        self.ca = CoordAtt(c_ * 4, c_ * 4)

        # 如果输入输出通道一致，可以使用残差
        self.add = c1 == c2

    def forward(self, x):
        x_in = self.cv1(x)
        y1 = self.m(x_in)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # 拼接四个尺度的特征
        y = torch.cat([x_in, y1, y2, y3], 1)

        # 应用注意力机制增强关键特征
        y = self.ca(y)

        # 通道压缩与输出
        out = self.cv2(y)

        return out + x if self.add else out

class AS_SPPF(nn.Module):
    """
    Adaptive Strip-Dilated SPPF (AS-SPPF)
    旨在通过空洞卷积和条带池化提升 YOLO11 对复杂背景和多尺度目标的感知力。
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

        # 1. 传统池化分支 (捕捉局部多尺度)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 2. 空洞卷积分支 (扩大感受野，不丢失空间分辨率)
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(c_, c_, 3, padding=2, dilation=2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # 3. 条带池化分支 (针对长条形目标优化)
        self.strip_h = nn.AdaptiveAvgPool2d((None, 1))
        self.strip_w = nn.AdaptiveAvgPool2d((1, None))
        self.strip_conv = nn.Conv2d(c_, c_, 1, 1, bias=False)

        # 4. 动态权重融合
        self.weight_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_ * 4, 4, 1, 1, bias=False),  # 4个分支的权重
            nn.Softmax(dim=1)
        )

        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        x = self.act(self.bn1(self.cv1(x)))

        # 串行池化
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)

        # 稀疏特征提取
        y3 = self.dilated_conv(x)

        # 条带上下文提取
        h, w = x.shape[2:]
        sh = self.strip_h(x).expand(-1, -1, h, w)
        sw = self.strip_w(x).expand(-1, -1, h, w)
        y4 = self.strip_conv(sh + sw)

        # 拼接特征
        feats = torch.cat([x, y2, y3, y4], dim=1)

        # 计算动态权重并应用 (可选，如果追求极致性能可直接concat后接cv2)
        # weights = self.weight_conv(feats)
        # feats = feats * weights # 简单示意，实际需对齐维度

        return self.act(self.bn2(self.cv2(feats)))


class MixPool(nn.Module):
    """混合池化层：融合 MaxPool 和 AvgPool 的信息"""

    def __init__(self, k):
        super().__init__()
        self.max = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avg = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        return self.max(x) + self.avg(x)


class LKA(nn.Module):
    """
    Large Kernel Attention (LKA) based on VAN:
    通过卷积分解模拟超大感受野 (相当于 21x21)，精度极高。
    """

    def __init__(self, dim):
        super().__init__()
        # 1. 局部特征提取
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 2. 空间长程依赖 (通过空洞卷积模拟超大核)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 3. 通道交互
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        # 动态加权
        return u * attn


class GatedGCBlock(nn.Module):
    """Gated Global Context Block: 加入门控机制的全局上下文"""

    def __init__(self, in_channels, ratio=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = int(in_channels * ratio)
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # 门控分支
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, in_channels, kernel_size=1)
        )
        # 动态门控
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # Context Modeling
        input_x = input_x.view(batch, channel, height * width)
        mask = self.conv_mask(x).view(batch, 1, height * width)
        mask = self.softmax(mask)
        context = torch.matmul(input_x, mask.transpose(1, 2))  # [B, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        # Transform & Fusion with Gating
        out = self.channel_add_conv(context)
        # 门控筛选有用全局特征
        g = self.gate(out)
        return x + out * g


class DC_SPPF(nn.Module):
    """
    Discriminative-Contextual SPPF (DC-SPPF)
    基于 HCD_SPPF 的终极精度增强版。
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, 1, 1), nn.BatchNorm2d(c_), nn.SiLU())

        # 1. 增强型多尺度池化路径 (MixPool)
        self.m = MixPool(k)

        # 2. LKA 注意力路径 (替代原始 $7\times 7$)
        self.lka = LKA(c_)

        # 3. Gated 全局上下文路径
        self.gated_gc = GatedGCBlock(c_)

        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 4. 多尺度空间注意力精炼
        self.multi_scale_att = nn.Sequential(
            nn.Conv2d(c_ * 4, c_ * 4, kernel_size=3, padding=1, groups=c_ * 4),
            nn.Conv2d(c_ * 4, c_ * 4, kernel_size=3, padding=2, dilation=2, groups=c_ * 4),
            nn.Conv2d(c_ * 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cv1(x)

        # 判别式特征分支并行提取
        y1 = self.m(x)  # 混合池化特征1
        y2 = self.m(y1)  # 混合池化特征2
        y3 = self.lka(x)  # LKA 形状特征
        y4 = self.gated_gc(x)  # 筛选后的全局特征

        # 特征拼接
        feat = torch.cat([y1, y2, y3, y4], 1)

        # 多尺度空间 Mask 精炼定位
        att = self.multi_scale_att(feat)
        feat = feat * att

        return self.cv2(feat)


class ConvGELU(nn.Module):
    """自定义 Conv-BN-GELU 层，替换 SiLU 以处理复杂特征。"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class FocalModulation(nn.Module):
    """简单动态焦点调制注意力，受 Focal Modulation 启发。"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # 动态调制

class C3TR(nn.Module):
    """Cross-Stage Partial Transformer 模块，用于全局注意力。"""
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
        y2 = y2.flatten(2).permute(0, 2, 1)  # 为 MHA 准备
        y2, _ = self.mha(y2, y2, y2)
        y2 = y2.permute(0, 2, 1).view(b, c, h, w)
        return self.cv3(torch.cat([y1, y2], 1))

class DynamicFocalSPPF(nn.Module):
    """重新设计的 SPPF v2，修复通道匹配，包含动态焦点调制、多尺度扩张卷积、Transformer 和多级 shortcut。"""

    def __init__(self, c1: int, c2: int, rates: tuple = (1, 2, 4)):
        """
        初始化 DynamicFocalSPPFv2 层。

        Args:
            c1 (int): 输入通道。
            c2 (int): 输出通道（通常 c2 == c1）。
            rates (tuple): 扩张卷积率。

        Notes:
            扩展 SPP(k=(3,5,7))，添加动态焦点调制以适应小目标、多尺度扩张卷积以丰富细节、C3TR 以捕获全局依赖，以及投影的多级 shortcut 以改善流动。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = ConvGELU(c1, c_, 1, 1)
        self.cv2 = ConvGELU(c_ * (len(rates) + 2), c2, 1, 1)  # +2 为 cv1 和 C3TR 输出
        self.dilated_convs = nn.ModuleList([nn.Conv2d(c_, c_, 3, 1, dilation=r, padding=r, bias=False) for r in rates])
        self.focal = FocalModulation(c_)
        self.c3tr = C3TR(c_, c_)
        self.mid_proj = ConvGELU(c_, c2, 1, 1) if c2 != c_ else nn.Identity()  # 投影 mid 到 c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用动态焦点调制、多尺度扩张卷积、Transformer 并返回拼接特征图与投影的多级 shortcut。"""
        y = [self.cv1(x)]  # 从 cv1 输出开始
        mid = y[-1]  # 中间 shortcut
        for dc in self.dilated_convs:  # 多尺度扩张卷积
            dilated = dc(mid)
            focaled = self.focal(dilated)  # 动态焦点调制
            y.append(focaled)
        tr_out = self.c3tr(mid)  # Transformer 增强
        y.append(tr_out)
        cat_features = torch.cat(y, 1)  # 拼接：形状兼容 (c_ * (len(rates) + 2))
        output = self.cv2(cat_features)
        mid_proj = self.mid_proj(mid)  # 投影 mid
        return output + mid_proj + x  # 多级残差 shortcut（现在通道匹配，假设 c2 == c1）
class ViTPatchEmbed(nn.Module):
    """ViT 风格的 Patch Embedding。"""
    def __init__(self, c1: int, patch_size: int = 4, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(c1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, D
        x = self.norm(x)
        return x, (h // 4, w // 4)  # 返回嵌入和原始 H/W 以恢复

class MultiHeadAttention(nn.Module):
    """多头自注意力层。"""
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x, _ = self.mha(x, x, x)
        return x + res

class CapsuleLayer(nn.Module):
    """简单动态路由胶囊层。"""
    def __init__(self, in_caps: int, out_caps: int, in_dim: int, out_dim: int, routings: int = 3):
        super().__init__()
        self.routings = routings
        self.W = nn.Parameter(torch.randn(out_caps, in_caps, in_dim, out_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: B, in_caps, in_dim
        b, in_caps, in_dim = u.shape
        out_caps = self.W.size(0)
        u_hat = torch.matmul(u[:, None, :, None, :], self.W[None, :, :, :, :])  # B, out_caps, in_caps, out_dim
        b_ij = torch.zeros(b, out_caps, in_caps, 1, device=u.device)
        for _ in range(self.routings):
            c_ij = torch.softmax(b_ij, dim=1)
            s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
            v_j = self.squash(s_j)
            if _ < self.routings - 1:
                a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + a_ij
        return v_j.squeeze(2)

    def squash(self, s: torch.Tensor) -> torch.Tensor:
        sq = torch.sum(s ** 2, dim=-1, keepdim=True)
        return (sq / (1 + sq)) * (s / torch.sqrt(sq + 1e-8))

class ViTEnhancedSPPF(nn.Module):
    """重新设计的 SPPF，包含 ViT patch embedding、多头注意力、动态胶囊和多级 shortcut。"""

    def __init__(self, c1: int, c2: int, patch_size: int = 4, embed_dim: int = 256, num_heads: int = 8):
        """
        初始化 ViTEnhancedSPPF 层。

        Args:
            c1 (int): 输入通道。
            c2 (int): 输出通道（通常 c2 == c1）。
            patch_size (int): ViT patch 大小。
            embed_dim (int): 嵌入维度。
            num_heads (int): 注意力头数。

        Notes:
            整合 ViT embedding 以全局捕捉、多头注意力以长距离依赖、胶囊以动态路由，以及多级 shortcut 以改善流动。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = ConvGELU(c1, c_, 1, 1)
        self.patch_embed = ViTPatchEmbed(c_, patch_size, embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.capsule = CapsuleLayer(1, 1, embed_dim, embed_dim)  # 简单胶囊用于聚合
        self.cv2 = ConvGELU(embed_dim, c2, 1, 1)  # 恢复通道
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用 ViT embedding、注意力、胶囊并返回特征图与多级 shortcut。"""
        cv1_out = self.cv1(x)  # 中间 shortcut 1
        patches, (ph, pw) = self.patch_embed(cv1_out)
        attn_out = self.mha(patches)  # 中间 shortcut 2
        caps_out = self.capsule(attn_out[:, None, :])  # 动态路由 (B, 1, D) -> (B, D)
        caps_out = self.norm(caps_out)
        # 恢复空间维度
        b, d = caps_out.shape
        restored = caps_out.view(b, d, ph, pw)
        output = self.cv2(restored)
        return output + cv1_out + x  # 多级残差 shortcut（假设 c2 == c1）



