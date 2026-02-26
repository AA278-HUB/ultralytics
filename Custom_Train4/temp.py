import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


# ==================== 测试代码 ====================
if __name__ == "__main__":
    C = 32
    K = 11
    model = DilatedReparamConv(c=C, k=K)
    model.eval()  # 必须先设为 eval 模式保证 BN 统计量固定

    x = torch.randn(1, C, 64, 64)
    out_train = model(x)

    # 执行融合
    model.switch_to_deploy()
    out_deploy = model(x)

    # 检查误差
    diff = (out_train - out_deploy).abs().mean()
    print(f"Mean Difference: {diff.item():.2e}")  # 通常在 1e-6 左右