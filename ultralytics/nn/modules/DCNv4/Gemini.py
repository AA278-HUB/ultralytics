import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------
# 1. 基础组件 (保留 Ultralytics 风格)
# ------------------------------------------------------

class Conv(nn.Module):
    '''Standard Conv'''

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ------------------------------------------------------
# 2. 改进核心 A: Coordinate Attention (替代普通 SE/PSA)
# ------------------------------------------------------
class CoordAtt(nn.Module):
    """
    Coordinate Attention: 相比全局平均池化，它保留了位置信息 (CVPR SOTA)
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 保留 H 维度
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 保留 W 维度

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # YOLO 常用 SiLU

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 两个方向分别池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接处理 (减少计算量)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 双向加权
        out = identity * a_h * a_w
        return out


# ------------------------------------------------------
# 3. 改进核心 B: Star-Style Large Kernel Block
# ------------------------------------------------------
class StarBlock(nn.Module):
    """
    StarNet Style Block + Large Kernel
    特点：使用 f(x) * g(x) 的高阶交互，无需激活函数，隐式提升维度
    """

    def __init__(self, dim, kersize=7, mlp_ratio=2.0, drop_path=0.):
        super().__init__()
        # 1. 深度卷积 (Large Kernel) - 捕捉长距离依赖
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kersize, padding=kersize // 2, groups=dim, bias=False)
        self.bn_dw = nn.BatchNorm2d(dim)

        # 2. 宽扩展 (Expansion)
        mid_dim = int(dim * mlp_ratio)
        self.f1 = nn.Conv2d(dim, mid_dim, 1, bias=False)  # Branch 1
        self.f2 = nn.Conv2d(dim, mid_dim, 1, bias=False)  # Branch 2
        self.bn_f = nn.BatchNorm2d(mid_dim)

        # 3. 输出投影
        self.g = nn.Conv2d(mid_dim, dim, 1, bias=True)
        self.act = nn.SiLU()  # 用于分支 Gating

    def forward(self, x):
        shortcut = x
        # 第一步：大核深度卷积
        x = self.dwconv(x)
        x = self.bn_dw(x)

        # 第二步：Star Operation (Element-wise Multiplication)
        # 类似 StarNet/ConvNeXt V2/GEGLU
        x1, x2 = self.f1(x), self.f2(x)

        # Star Interaction: (x1 * x2)
        # 这里是一个技巧：不需要ReLU，乘法本身产生了非线性。
        # 为了稳定，我们在 x2 上加个激活，构成 Gated Linear Unit
        x = x1 * self.act(x2)

        x = self.bn_f(x)

        # 第三步：投影回原维度
        x = self.g(x)

        return x + shortcut


# ------------------------------------------------------
# 4. 主模块: RepHMS-Star (融合上述改进)
# ------------------------------------------------------
class RepHMSStar(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, kersize=7,
                 expansion=0.5, depth_expansion=2):
        super(RepHMSStar, self).__init__()

        self.width = width
        self.depth = depth

        # 通道计算
        c_ = int(out_channels * expansion)  # Split 通道数
        c_hidden = int(c_ * depth_expansion)  # Block 内部膨胀

        # 1. 输入变换
        self.conv1 = Conv(in_channels, c_ * width, 1, 1)

        # 2. 堆叠 StarBlock (替代原来的 DepthBottleneck)
        self.blocks = nn.ModuleList()
        for _ in range(width - 1):
            # 使用 StarBlock，它比普通 Conv 具有更强的表征能力
            layers = nn.ModuleList([
                StarBlock(dim=c_, kersize=kersize, mlp_ratio=depth_expansion)
                for _ in range(depth)
            ])
            self.blocks.append(layers)

        # 3. 计算最终 concat 后的通道数
        total_channels = c_ * width  # Base split channels
        # 注意：RepHMS 的逻辑里，elan 列表会不断 append 新特征
        # 这里的通道计算需要和 forward 对齐
        # 初始 elan = [x[0]], 循环 width-1 次，每次 append 一个 block 的输出
        final_concat_channels = c_ + (c_ * (width - 1))

        # 4. 坐标注意力 (替代原来的 AdaptiveAvgPool)
        self.att = CoordAtt(final_concat_channels)

        # 5. 输出变换
        self.conv2 = Conv(final_concat_channels, out_channels, 1, 1)

    def forward(self, x):
        # Input Projection
        y = self.conv1(x)

        # Split (Chunk)
        # dynamic split based on width
        split_size = y.shape[1] // self.width
        x_out = list(torch.split(y, split_size, dim=1))

        # 初始残差连接
        x_out[1] = x_out[1] + x_out[0]

        cascade = []  # 级联缓存
        elan = [x_out[0]]  # 最终聚合列表

        for i in range(self.width - 1):
            # 处理级联加法 (Dense Connection)
            if i > 0 and len(cascade) > 0:
                # 简单的加法可能导致语义混淆，这里可以加一个加权
                # 但为了速度，保持 Add
                x_out[i + 1] = x_out[i + 1] + cascade[-1]

                # 通过深度模块 (StarBlock)
            for j in range(self.depth):
                x_out[i + 1] = self.blocks[i][j](x_out[i + 1])

            # 收集结果
            elan.append(x_out[i + 1])

            # 更新级联缓存
            cascade.append(x_out[i + 1])

        # Concat
        y_out = torch.cat(elan, 1)

        # Apply Coordinate Attention (位置敏感的注意力)
        y_out = self.att(y_out)

        # Final Projection
        y_out = self.conv2(y_out)

        return y_out


# ------------------------------------------------------
# 测试代码
# ------------------------------------------------------
if __name__ == "__main__":
    # 配置：大核设为 7，模拟 UniRepLKNet 的感受野
    model = RepHMSStar(in_channels=64, out_channels=256, width=3, depth=1, kersize=7)

    # 输入
    dummy = torch.randn(1, 64, 64, 64)

    # 推理
    output = model(dummy)

    print(f"RepHMS-Star Input: {dummy.shape}")
    print(f"RepHMS-Star Output: {output.shape}")

    # 简单验证参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    print("改进成功：引入了 StarNet 交互机制与 Coordinate Attention。")