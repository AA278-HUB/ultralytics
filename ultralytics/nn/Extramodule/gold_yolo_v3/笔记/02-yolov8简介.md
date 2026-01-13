**YOLOV8原理和实现全解析**可以参考MMYOLO的这篇文章

https://mmyolo.readthedocs.io/zh-cn/latest/recommended_topics/algorithm_descriptions/yolov8_description.html

然后咱这里只进行一个简单的梳理

# Backbone

![                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119164313497.png)

# Neck

![image-20240119165758605](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119165758605.png)

# 源码解析

这部分涉及源码主要就是tasks.py文件（D:\anaconda3\envs\GOLD-YOLO\Lib\site-packages\ultralytics\nn\tasks.py）

主要涉及的函数

解析：parse_model

```
def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    # 最大通道
    max_channels = float('inf')
    # 总类别 指定激活函数 所有模型大小
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    # 模型深度因子（和模块数量挂钩） 模型宽度因子（和通道数挂钩）
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        # 获取当前模型大小
        scale = d.get('scale')
        if not scale:
            # 未设定模型大小 默认设置n
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        # 如果设置了激活函数 将Conv模块中的激活函数设置为指定激活函数
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    # 每一层的输出通道
    ch = [ch]
    # layers用于存储每层的模块 save存储要保存的层索引 c2表示每层的索引参数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 遍历配置文件 i:索引 f:输入索引 n:重复次数 m:模块名 args:传入参数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 根据模块名得到对应模块
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        # 模型深度受到深度因子影响
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        # 因为不同模块的参数类型不同 所以按照参数分类
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
            # c1：输入通道 c2: 输出通道
            c1, c2 = ch[f], args[0]
            # 模型宽度（通道数）受到模型宽度因子影响
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            # 这一类模块的参数类型都是 (in_channel,out_channel, 剩余参数)
            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                # 在上面的基础上加上n 重复因子
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            # 输入通道+剩余参数
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            # 参数就是输入通道
            args = [ch[f]]
        elif m is Concat:
            # 没有参数
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose):
            # head层 将通道数做一个累加
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        # 如果n>1 封装成一个序列模块
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 获取模块类型
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 获取模块参数
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        # 配置当前模块的信息
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        # 输出模块信息
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        # 将不为-1的输入 保存到save列表
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将当前模块添加到layers层
        layers.append(m_)
        # 第一层 由于之前初始化 ch=[ch] 做一个重置操作
        if i == 0:
            ch = []
        # 添加每一层的输出层通道数
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
```

执行：_predict_once

```
      def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        # y是个列表 根据self.save(要存储的对象)存储每层的输出
        y, dt = [], []  # outputs
        # 遍历构建好的模型
        for m in self.model:
            # 输入不为前一层
            if m.f != -1:  # if not from previous layer
                # if isinstance(m.f, int): y[m.f] 单层且输入不为前一层 从y中找输入
                # else [x if j == -1 else y[j] for j in m.f]  else 只能是多层输入 判断是否输入是-1 如果是-1 对应输入为x, 否则从y中查找
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            # 前向传播
            x = m(x)  # run
            # 如果在保存列表 保存在y中 否则y中保存None
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
```



# 附件

（1）YOLOv8-P5 模型结构

![image-20240119160935929](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119160935929.png)

（2）yolov8.yaml文件

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```





