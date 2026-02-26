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
        c_ = int(c2 * e*2)
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


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