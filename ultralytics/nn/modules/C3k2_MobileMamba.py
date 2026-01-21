import torch
from torch import nn

from ultralytics.nn.Extramodule.MambaVision import MambaVisionBlock
from ultralytics.nn.Extramodule.mobileMamba.mobilemamba import MobileMambaBlock
from ultralytics.nn.modules.block import C3k, C3k2


class C3k_MobileMamba(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MobileMambaBlock(c_) for _ in range(n)))
class C3k2_MobileMamba(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MobileMamba(self.c, self.c, n, shortcut, g) if c3k else MobileMambaBlock(self.c) for _ in range(n))


######################################## CVPR2025 MambaVision start ########################################

class C3k_MambaVision(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MambaVisionBlock(c_) for _ in range(n)))


class C3k2_MambaVision(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_MambaVision(self.c, self.c, 2, shortcut, g) if c3k else MambaVisionBlock(self.c) for _ in range(n))


######################################## CVPR2025 MambaVision end ########################################

if __name__ == "__main__":
        # 设置随机种子，方便复现
        torch.manual_seed(0)

        # 构造一个假的输入特征图
        x = torch.randn(1, 64, 80, 80)  # (B, C, H, W)

        print("Input shape:", x.shape)

        # ===============================
        # 测试 C3k_MobileMamba
        # ===============================
        model1 = C3k2_MambaVision(
            c1=64,
            c2=128,
            # n=2,
            # shortcut=True
        )

        y1 = model1(x)
        print("C3k2_MambaVision output shape:", y1.shape)

        # ===============================
        # 测试 C3k2_MobileMamba
        # ===============================
        # model2 = C3k2_MobileMamba(
        #     c1=64,
        #     c2=128,
        #     n=2,
        #     c3k=True
        # )
        #
        # y2 = model2(x)
        # print("C3k2_MobileMamba output shape:", y2.shape)
        #
        # print("✅ Forward test passed.")