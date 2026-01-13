# 1.准备工作

GOLD-YOLO论文建议大家先读个2遍左右，有个基础认识。论文地址：https://arxiv.org/pdf/2309.11331.pdf

# 2. 论文简介

GOLD-YOLO主要进行了Neck部分的修改。作者提出了聚集和分发机制（Gather-and-Distribute mechanism，GD）。

GD主要包含三个模块：特征对齐模块 (Feature Alignment Module，FAM)，信息融合模块(Information Fusion Module，IFM)和信息注入模块(Information Injection Module，Inject)。为了增强模型对不同大小的物体的检测能力，作者又设计了两个分支：浅层GD (low-stage gather-and-distribute branch，Low-GD)和深层GD (high-stage gather-and-distribute branch，High-GD)。

## 2.1 Low-GD

![image-20240121101947377](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121101947377.png)

然后信息注入这里咱演示一下Inject_P3，Inject_P4是类似的。

![image-20240121093820046](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121093820046.png)

**LOW-GD操作完成，得到输出P3:80\*80\*256，P4:40\*40\*512**

## 2.2 High-GD

![image-20240121123850904](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121123850904.png)



![image-20240122191939595](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240122191939595.png)



**HIGH-GD操作完成，得到Neck层的输N3:80\*80\*256，N4:40\*40\*512，N5:20\*20\*1024**





