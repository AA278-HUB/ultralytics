# 1.源码地址

GOLD-YOLO源码地址：https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO

# 2.源码介绍

configs下的配置文件及对应不同模型大小的GOLD-YOLO网络，咱这里以n模型为准进行讲解。

![image-20240121095724075](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121095724075.png)

咱只关注neck部分就行了

```
neck=dict(
                type='RepGDNeck',
                num_repeats=[12, 12, 12, 12],
                out_channels=[256, 128, 128, 256, 256, 512],
                extra_cfg=dict(
                        norm_cfg=dict(type='SyncBN', requires_grad=True),
                        depths=2,
                        fusion_in=480,
                        fusion_act=dict(type='ReLU'),
                        fuse_block_num=3,
                        embed_dim_p=96,
                        embed_dim_n=352,
                        key_dim=8,
                        num_heads=4,
                        mlp_ratios=1,
                        attn_ratios=2,
                        c2t_stride=2,
                        drop_path_rate=0.1,
                        trans_channels=[64, 32, 64, 128],
                        pool_mode='torch'
                )
        )
```

neck部分可以看到neck部分的主体类是RepGDNeck，Ctrl+n(或者两下shift搜索这个类进入)

咱需要分别对照init模块的初始化 和 forward前向传播过程中来看。咱按照我们之前讲解的结构来一个一个看。

## 2.1 Low_FAM

\__init__ 部分

```
self.low_FAM = SimFusion_4in()
```

forward部分

```
# 这里将参数名改了下 对照咱之前的图模型
(B2, B3, B4, B5) = input

# Low_GD
## use conv fusion global info
low_align_feat = self.low_FAM(input)
```

SimFusion_4in()

```
class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        # 池化
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        # 接收四个输入 B2 B3 B4 B5
        x_l, x_m, x_s, x_n = x
        # 要对齐的模型B4的shape
        B, C, H, W = x_s.shape
        # 对齐的尺寸size
        output_size = np.array([H, W])
        
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        
        # B2 B3进行池化操作
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        # B5 进行线性插值操作
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        # 将对齐结果Concat后返回
        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out
```

## 2.2 Low_IFM

\__init__ 部分

```
self.low_IFM = nn.Sequential(
				# extra_cfg.fusion_in 输入通道  extra_cfg.embed_dim_p 中间通道
				Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
				# RepConv blocks
                *[block(extra_cfg.embed_dim_p, extra_cfg.embed_dim_p) for _ in 														range(extra_cfg.fuse_block_num)],
                # 通道数转换:卷积将通道数转为B4和B3对应的通道
                # # extra_cfg.embed_dim_n=96  sum(extra_cfg.trans_channels[2:4])=64+32
                Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
```

forward部分

```
low_fuse_feat = self.low_IFM(low_align_feat)
# low_global_c4 + low_global_c3
low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1) 
```

block：RepVGGBlock （这个块有兴趣的话你可以自己看看，我就不过多介绍了）

## 2.3 LOW_Inject（P4注入）

\__init__ 部分

```
# SimConv其实这个就是一个简单的CBA结构，这里kernel_size和stride都设置为1用于转换通道数
self.reduce_layer_c5 = SimConv(
        in_channels=channels_list[4],  # 1024
        out_channels=channels_list[5],  # 512
        kernel_size=1,
        stride=1
)
# LAF融合邻层特征
self.LAF_p4 = SimFusion_3in(
        in_channel_list=[channels_list[3], channels_list[3]],  # 512, 256
        out_channels=channels_list[5],  # 256
)
# 进行Inject注入
self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[5], channels_list[5], norm_cfg=extra_cfg.norm_cfg,activations=nn.ReLU6)
# 最后通过一个RepBlock
self.Rep_p4 = RepBlock(
        in_channels=channels_list[5],  # 512
        out_channels=channels_list[5],  # 512
        n=num_repeats[5],
        block=block
)
```

forward 部分

```
c5_half = self.reduce_layer_c5(c5)
p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
p4 = self.Rep_p4(p4)
```

咱先来来看LAF操作

![image-20240121112554827](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121112554827.png)



然后咱再看Inject注入过程

![image-20240121113835575](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121113835575.png)

##                2.4 LOW_Inject（P3注入）

\__init__ 部分

```
# SimConv其实这个就是一个简单的CBA结构，这里kernel_size和stride都设置为1用于转换通道数
self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[6],  # 128
                kernel_size=1,
                stride=1
        )
# LAF融合邻层特征
self.LAF_p3 = SimFusion_3in(
        in_channel_list=[channels_list[5], channels_list[5]],  # 512, 256
        out_channels=channels_list[6],  # 256
)
# 进行Inject注入
self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[6], channels_list[6], norm_cfg=extra_cfg.norm_cfg,activations=nn.ReLU6)
# 最后通过一个RepBlock
self.Rep_p3 = RepBlock(
        in_channels=channels_list[6],  # 128
        out_channels=channels_list[6],  # 128
        n=num_repeats[6],
        block=block
)
```

forward部分    

```
# p4通道数的一个减半  注意这里是p4 不是c4
p4_half = self.reduce_layer_p4(p4)
# LAF融合邻层特征
p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
p3 = self.Rep_p3(p3)
```

LAF:

![image-20240121131640546](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121131640546.png)

Inject

![image-20240121132012336](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121132012336.png)

## 2.5 HIGH_FAM

\__init__ 部分

```
# extra_cfg.c2t_stride=2 extra_cfg.pool_mode="torch"
self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
```

forward部分                                                                

```
high_align_feat = self.high_FAM([p3, p4, c5])        
```

PyramidPoolAgg

```
class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
    
    def forward(self, inputs):
    	# inputs:[p3, p4, c5] 得到c5的BCHW
        B, C, H, W = get_shape(inputs[-1])
        # 计算目标尺寸
        # H = (H - 1) // self.stride + 1
        # W = (W - 1) // self.stride + 1
        
        # 使用c5的HW
        output_size = np.array([H, W])
        
        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d
        
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        
        # 做池化
        out = [self.pool(inp, output_size) for inp in inputs]
        # 按通道连接
        return torch.cat(out, dim=1)
```

## 2.5 HIGH_IFM

\__init__ 部分

```
# extra_cfg.drop_path_rate=0.1 extra_cfg.depths=2
dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
self.high_IFM = TopBasicLayer(
		# 2
        block_num=extra_cfg.depths,
        # 352
        embedding_dim=extra_cfg.embed_dim_n,
        # 8
        key_dim=extra_cfg.key_dim,
        # 4
        num_heads=extra_cfg.num_heads,
        # 1
        mlp_ratio=extra_cfg.mlp_ratios,
        # 2
        attn_ratio=extra_cfg.attn_ratios,
        drop=0, attn_drop=0,
        drop_path=dpr,
        # norm_cfg=dict(type='SyncBN', requires_grad=True)
        norm_cfg=extra_cfg.norm_cfg
)
# 变换通道数
# extra_cfg.embed_dim_n=352  sum(extra_cfg.trans_channels[2:4])=64+128
# high_global_p4 + low_global_p5
self.conv_1x1_n = nn.Conv2d(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), 1, 1, 0)
```

forward部分  

```
high_fuse_feat = self.high_IFM(high_align_feat)
high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
```

## 2.6 HIGH_Inject(N4注入) 

\__init__ 部分

```
# 池化对齐
self.LAF_n4 = AdvPoolFusion()
# 信息注入
self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[8], channels_list[8],
                                             norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
# RepBlock
self.Rep_n4 = RepBlock(
        in_channels=channels_list[6] + channels_list[7],  # 128 + 128
        out_channels=channels_list[8],  # 256
        n=num_repeats[7],
        block=block
)
```

forward部分   

```
n4_adjacent_info = self.LAF_n4(p3, p4_half)
n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
n4 = self.Rep_n4(n4)
```

咱先看LAF模块

（下面reduce_layer_p5名称写错了 应该是reduce_layer_p4）

![image-20240121130733301](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121130733301.png)

然后咱再来看看Inject

![image-20240121130804764](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121130804764.png)’

## 2.7 HIGH_Inject(N5注入) 

\__init__ 部分

```
# 池化对齐
self.LAF_n5 = AdvPoolFusion()
# 信息注入
self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[10], channels_list[10],
                                             norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
# RepBlock                                            
self.Rep_n5 = RepBlock(
        in_channels=channels_list[5] + channels_list[9],  # 256 + 256
        out_channels=channels_list[10],  # 512
        n=num_repeats[8],
        block=block
)
```

forward部分 

```
n5_adjacent_info = self.LAF_n5(n4, c5_half)
n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
n5 = self.Rep_n5(n5)
```

咱先看LAF模块

![image-20240121132539409](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121132539409.png)                    

然后咱再来看看Inject                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

![image-20240121132551689](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121132551689.png)







