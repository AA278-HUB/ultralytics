import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np

class RepGhostModule(nn.Module):
    """
    RepGhost module with training and deploy modes.
    This module can fuse convolutions during inference for efficiency.
    """
    default_act = nn.ReLU(inplace=True)
    def __init__(self, c1, c2, kernel_size=1, dw_size=3, stride=1, relu=True, reparam_bn=True, reparam_identity=False, deploy=False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.stride = stride
        self.act = self.default_act if relu else nn.Identity()
        self.deploy = deploy
        # 主卷积分支（移除激活函数）
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(c2),
        )
        # 可融合的分支列表
        fusion_conv, fusion_bn = [], []
        if reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(c2))
        if reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())
        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)
        # cheap 卷积分支（深度卷积）
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c2, dw_size, 1, dw_size // 2, groups=c2, bias=False),
            nn.BatchNorm2d(c2)
        )
    def forward(self, x):
        if self.deploy:
            return self.act(self.conv(x))
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.act(x2)
    def forward_fuse(self, x):
        """推理模式前向"""
        return self.act(self.conv(x))
    def get_equivalent_kernel_bias(self):
        """计算分支的等效卷积核和偏置"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3
    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            # Identity卷积
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)
        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)
    def fuse_convs(self):
        """融合所有卷积，生成单一卷积用于推理"""
        if hasattr(self, "conv"):
            return
        # 获取分支的融合内核和偏置
        branch_kernel, branch_bias = self.get_equivalent_kernel_bias()
        # 融合主分支conv1
        pw_kernel, pw_bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        # 计算整体等效内核和偏置
        full_kernel = pw_kernel * branch_kernel  # [c2, c1, 1, 1] * [c2, 1, 3, 3] -> [c2, c1, 3, 3]
        dw_sum = branch_kernel.sum(dim=(2, 3))  # [c2, 1]
        full_bias = pw_bias * dw_sum.squeeze() + branch_bias
        # 创建融合后的卷积（groups=1）
        self.conv = nn.Conv2d(
            in_channels=self.c1,
            out_channels=self.c2,
            kernel_size=self.conv2[0].kernel_size,
            stride=self.stride,
            padding=self.conv2[0].padding,
            bias=True
        ).requires_grad_(False)
        self.conv.weight.data = full_kernel
        self.conv.bias.data = full_bias
        # 设置部署模式
        self.deploy = True
        # 删除训练用分支
        for attr in ["conv1", "conv2", "fusion_conv", "fusion_bn"]:
            if hasattr(self, attr):
                self.__delattr__(attr)
torch.manual_seed(0)
module = RepGhostModule(3, 6, kernel_size=1, dw_size=3, stride=1, relu=True, reparam_bn=True)
module.eval()
x = torch.randn(1, 3, 32, 32)
y_train = module(x)
module.fuse_convs()
y_fuse = module(x)
print(torch.allclose(y_train, y_fuse, atol=1e-5))  # True