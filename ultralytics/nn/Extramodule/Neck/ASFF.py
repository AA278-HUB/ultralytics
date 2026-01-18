import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        # init asff_module = ASFF(level=1, multiplier=1, rfb=False, vis=False)
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]
        # print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(512 * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        # forward output_feature = asff_module([level_2_feature, level_1_feature, level_0_feature])
        x_level_0 = x[2]  # 最大特征层 level_0_feature = (1, 1024, 20, 20)  # 大尺寸特征图 尺寸小通道多
        x_level_1 = x[1]  # 中间特征层 level_1_feature = (1, 512, 40, 40)   # 中尺寸特征图
        x_level_2 = x[0]  # 最小特征层 level_2_feature = (1, 256, 80, 80)   # 小尺寸特征图

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(
                x_level_0)  # (1, 1024, 20, 20) → self.compress_level_0 =  Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False) → (1, 512, 20, 20)

            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2,
                mode='nearest')  # (1, 512, 20, 20) → F.interpolate→ [1, 512, 40, 40]
            level_1_resized = x_level_1  # [1, 512, 40, 40] → = → [1, 512, 40, 40]
            level_2_resized = self.stride_level_2(
                x_level_2)  # [1, 256, 80, 80] → self.stride_level_2 = Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) → [1, 512, 40, 40]
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(
            level_0_resized)  # [1, 512, 40, 40] → self.weight_level_0 = (conv): Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1), bias=False) → [1, 16, 40, 40]
        level_1_weight_v = self.weight_level_1(
            level_1_resized)  # [1, 512, 40, 40] → self.weight_level_1 = Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1), bias=False) → [1, 16, 40, 40]
        level_2_weight_v = self.weight_level_2(
            level_2_resized)  # [1, 512, 40, 40] → self.weight_level_2 = Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1), bias=False) → [1, 16, 40, 40]

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v),
            1)  # [1, 16, 40, 40],[1, 16, 40, 40],[1, 16, 40, 40] → cat → [1, 48, 40, 40]
        levels_weight = self.weight_levels(
            levels_weight_v)  # [1, 48, 40, 40] → self.weight_levels = (conv): Conv2d(48, 3, kernel_size=(1, 1), stride=(1, 1), bias=False) → [1, 3, 40, 40]
        levels_weight = F.softmax(levels_weight, dim=1)  # [1, 3, 40, 40] → F.softmax → [1, 3, 40, 40]

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]
        # [1, 512, 40, 40] * [1, 1, 40, 40] + [1, 512, 40, 40] * [1, 1, 40, 40] + [1, 512, 40, 40] * [1, 1, 40, 40] →  [1, 512, 40, 40]
        out = self.expand(
            fused_out_reduced)  # [1, 512, 40, 40] → self.expand = (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) → [1, 512, 40, 40]

        if self.vis:  # self.vis = False
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


if __name__ == "__main__":
    # 模拟的输入特征图，模拟三个不同尺度的特征图，例如来自一个多尺度特征提取网络的输出
    level_0_feature = torch.randn(1, 1024, 20, 20)  # 大尺寸特征图
    level_1_feature = torch.randn(1, 512, 40, 40)  # 中尺寸特征图
    level_2_feature = torch.randn(1, 256, 80, 80)  # 小尺寸特征图

    # 初始化ASFF模块，level表示当前ASFF模块处理的是哪个尺度的特征层，这里以处理中尺寸特征层为例
    # multiplier用于调整通道数，rfb和vis分别表示是否使用更丰富的特征表示和是否可视化
    asff_module = ASFF(level=1, multiplier=1, rfb=False, vis=False)

    # 通过ASFF模块传递特征图
    output_feature = asff_module([level_2_feature, level_1_feature, level_0_feature])

    # 打印输出特征图的形状，确保ASFF模块正常工作
    print(f"Output feature shape: {output_feature.shape}")

# TODO 计算流程图、原文框架图、公式表示