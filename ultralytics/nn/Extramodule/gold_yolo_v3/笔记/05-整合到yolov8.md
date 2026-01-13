咱先来了解一下YOLOV8的修改的一个基本过程

![image-20240121134327270](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121134327270.png)

![image-20240121134614058](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121134614058.png)

*  \__init__:注册专用
* block：块模型Block modules（例如：C2f、SPP）
* conv：卷积模型Convolution modules（例如：Conv、DWConv）
* head：模型的head头部分（例如：Detect）
* transformer：Transformer模块（例如：TransformerLayer）
* utils：模型工具类

那么咱的话新建一个gold_yolo.py来存储我们自己定义的GOLD-YOLO模块

写入咱的一个基础模块

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
GOLD-YOLO modules
"""

__all__ = ()
```

咱按顺序来一个一个添加

# 1.添加模块(主要根据GOLD-YOLO neck源码部分\__init__初始化函数)

我这里先把待会儿要用到的类先提前都声明出来了，正常的话是每添加模块，然后咱就声明这个模块。

```
__all__ = ('Low_FAM', 'Low_IFM', 'Split', 'SimConv', 'Low_LAF', 'Inject', 'RepBlock', 'High_FAM', 'High_IFM', 'High_LAF')
```

## Low_FAM

GOLD-YOLO源码

```
self.low_FAM = SimFusion_4in()
```

咱就把这个SimFusion_4in的类及其依赖的类添加进来 并把类名称改为Low_FAM.还得记得把对应的库跟着导入进来。

```
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
```

```
class Low_FAM(nn.Module):
    def __init__(self):
        super().__init__()
        # 池化
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        # 接收四个输入 B5 B4 B3 B2
        x_l, x_m, x_s, x_n = x
        #
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out

def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x
```

## Low_IFM

GOLD-YOLO源码

```
self.low_IFM = nn.Sequential(
                Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(extra_cfg.embed_dim_p, extra_cfg.embed_dim_p) for _ in range(extra_cfg.fuse_block_num)],
                Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
```

咱这里把这个改写一下

```
class Low_IFM(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim_p, fuse_block_num):
        super().__init__()
        self.conv1x1_1 = Conv(in_channels, embed_dim_p, kernel_size=1, stride=1, padding=0)
        self.block = nn.ModuleList([RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)]) 
        self.conv1x1_2 = Conv(embed_dim_p, out_channels, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, x):
        x = self.conv1x1_1(x)
        for block in self.block:
            x = block(x)
        out = self.conv1x1_2(x)
        return out

class Conv(nn.Module):
    '''Normal Conv with SiLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        
        else:
            self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
                                   
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
```

## Split

```
class Split(nn.Module):
    def __init__(self, trans_channels):
        super().__init__()
        self.trans_channels = trans_channels

    def forward(self, x):
        return x.split(self.trans_channels, dim=1)
```

## LOW_Inject（P4注入)

GOLD-YOLO源码

这里涉及三个模块:SimConv、Low_LAF、Inject、RepBlock

```
# SimConv
self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[4],  # 1024
                out_channels=channels_list[5],  # 512
                kernel_size=1,
                stride=1
        )
# Low_LAF
self.LAF_p4 = SimFusion_3in(
        in_channel_list=[channels_list[3], channels_list[3]],  # 512, 256
        out_channels=channels_list[5],  # 256
)
# Inject
self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[5], channels_list[5], norm_cfg=extra_cfg.norm_cfg,
                                             activations=nn.ReLU6)
# RepBlock
self.Rep_p4 = RepBlock(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[5],  # 256
                n=num_repeats[5],
                block=block
        )
```

**SimConv**

```
class SimConv(nn.Module):
    '''Normal Conv with ReLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))
```

**low_LAF**

```
#  改下名
class Low_LAF(nn.Module):
	# 输入通道只用到一个 改一下
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channels, out_channels, 1, 1)
        # self.cv_fuse = SimConv(out_channels * 3, out_channels, 1, 1)
        # 这里动了源码因为结构不匹配
        self.cv_fuse = SimConv(out_channels * 4, out_channels * 2, 1, 1)
        self.downsample = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        N, C, H, W = x[1].shape
        # 输出形状
        output_size = (H, W)
        
        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])

        x0 = self.downsample(x[0], output_size)
        x1 = self.cv1(x[1])
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))
```

**Inject**

note:这里Inject中多加了一个参数 因为接收的输入是从Split模块过来的 是一个全举特征列表 需要指定当前引用的是哪一个

```
from mmcv.cnn import ConvModule
```

```
# Inject
class Inject(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_index: int,
            norm_cfg_type='BN'
    ) -> None:
        super().__init__()
        self.norm_cfg = dict(type=norm_cfg_type, requires_grad=True)
        global_inp = inp
        self.global_index = global_index

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid() 
    
    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        # 获取对应索引的全局特征
        x_g = x_g[self.global_index]
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H
        
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)
        
        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        
        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + global_feat
        return out
        
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6
        
def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool
```

**RepBlock**

```
class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()
        
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None	
        '''
        # 这里没用到BottleRep 直接就注释掉了
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(
                    *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in
                      range(n - 1))) if n > 1 else None
        '''
    
    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x
        
class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        
        else:
            self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
    
    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
```

## LOW_Inject（P3注入）

和上面P4注入涉及的模块完全一样，就不用引入对应模块了。

```
self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[6],  # 128
                kernel_size=1,
                stride=1
        )
self.LAF_p3 = SimFusion_3in(
        in_channel_list=[channels_list[5], channels_list[5]],  # 512, 256
        out_channels=channels_list[6],  # 256
)
self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[6], channels_list[6], norm_cfg=extra_cfg.norm_cfg,
                                             activations=nn.ReLU6)
self.Rep_p3 = RepBlock(
        in_channels=channels_list[6],  # 128
        out_channels=channels_list[6],  # 128
        n=num_repeats[6],
        block=block
)
```

## HIGH_FAM

GOLD-YOLO源码

```
self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
```

直接改名就行

```
class High_FAM(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
    
    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        '''
        这里记得注释掉
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        '''
        
        output_size = np.array([H, W])
        
        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d
        
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        
        out = [self.pool(inp, output_size) for inp in inputs]
        
        return torch.cat(out, dim=1)

def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape
```

## HIGH_IFM

GOLD-YOLO源码

```
dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
self.high_IFM = TopBasicLayer(
        block_num=extra_cfg.depths,
        embedding_dim=extra_cfg.embed_dim_n,
        key_dim=extra_cfg.key_dim,
        num_heads=extra_cfg.num_heads,
        mlp_ratio=extra_cfg.mlp_ratios,
        attn_ratio=extra_cfg.attn_ratios,
        drop=0, attn_drop=0,
        drop_path=dpr,
        norm_cfg=extra_cfg.norm_cfg
)
```

咱得设计成一个模块

```
from mmcv.cnn import build_norm_layer
```

```
class High_IFM(nn.Module):
    def __init__(self, block_num, embed_dim_n, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path_rate=0.1,
                 depths=2, norm_cfg_type='BN2d',
                 act_layer=nn.ReLU6):
        super().__init__()
        drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        norm_cfg = dict(type=norm_cfg_type, requires_grad=True)
        self.block_num = block_num
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                    embed_dim_n, key_dim=key_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                    drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg, act_layer=act_layer))
               
    
    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x
    
class top_Block(nn.Module):
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,
                              norm_cfg=norm_cfg)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)
    
    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
                self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
    
    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k
        
        xx = torch.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx
    
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.add_module('c', nn.Conv2d(
                a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

## HIGH_Inject(N4注入) 

GOLG-YOLO源码

只有AdvPoolFusion模块是不同的 其他之前都定义过

```
self.LAF_n4 = AdvPoolFusion()
self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[8], channels_list[8],
                                             norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
self.Rep_n4 = RepBlock(
        in_channels=channels_list[6] + channels_list[7],  # 128 + 128
        out_channels=channels_list[8],  # 256
        n=num_repeats[7],
        block=block
)
```

```
class High_LAF(nn.Module):
    def forward(self, x1, x2):
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d
        
        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)
        
        return torch.cat([x1, x2], 1)
```

## HIGH_Inject(N5注入)

GOLG-YOLO源码

可以发现都定义过了

```
self.LAF_n5 = AdvPoolFusion()
self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[10], channels_list[10],
                                             norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
self.Rep_n5 = RepBlock(
        in_channels=channels_list[5] + channels_list[9],  # 256 + 256
        out_channels=channels_list[10],  # 512
        n=num_repeats[8],
        block=block
)

self.trans_channels = extra_cfg.trans_channels
```

# 2. 注册并配置模块

咱定义完之后咱得进行注册、引入和配置

## 注册

找到咱刚才编写的gold_yolo的同级目录下的\__init__.py

```
from .gold_yolo import (Low_FAM, Low_IFM, Split, SimConv, Low_LAF, Inject, RepBlock, High_FAM, High_IFM, High_LAF)
```

同时\__all__中添加

```
'Low_FAM', 'Low_IFM', 'Split', 'SimConv', 'Low_LAF', 'Inject', 'RepBlock', 'High_FAM', 'High_IFM', 'High_LAF'
```

![image-20240121153704683](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240121153704683.png)

## 引入

在tasks.py中引入，tasks.py在gold_yolo.py的上一级目录

在开头from ultralytics.nn.modules import中添加模块

```
, Low_FAM, Low_IFM, Split, SimConv, Low_LAF, Inject, RepBlock, High_FAM, High_IFM, High_LAF
```

## 配置

直接定位到这一行 ctrl+f 搜索定位

```
if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
```

然后咱一个一个来配置

### （1）Low_FAM

先看Low_FAM，先观察咱们刚才定义的Low_FAM

```
class Low_FAM(nn.Module):
    def __init__(self)
    def forward(self, x):
```

可以看到初始化不需要参数，并且输出通道数等于输入通道数之和，

然后咱往下找，看有没有满足条件的（即不需要参数，输出通道数为输入通道数之和）

发现刚好和Concat相同

所以将

```
elif m is Concat:
    # 输出通道数等于输入通道数之和，且初始化不需要参数
    c2 = sum(ch[x] for x in f)
```

改为

```
elif m in [Concat, Low_FAM]:
     # 没有参数
    c2 = sum(ch[x] for x in f)
```

### （2）Low_IFM

```
class Low_IFM(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim_p, fuse_block_num):
    def forward(self, x):
```

可以看到是输入参数是 	输入通道+输出通道+其他参数

可以看到第一个判断即满足条件 所以加上

```
if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, Low_IFM):
```

### （3）Split

```
class Split(nn.Module):
    def __init__(self, trans_channels):
    def forward(self, x):
```

需要参数 不需要输入通道数 但是输出有多个，也就是说输出通道有多个

在最后一个else上添加

```
elif m is Split:
    c2 = []
    for arg in args:
        if arg != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
            c2.append(make_divisible(min(arg, max_channels) * width, 8))
    args = [c2]
```

### （4）SimConv

```
class SimConv(nn.Module):
    '''Normal Conv with ReLU VAN_activation''' 
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
    def forward(self, x):
```

输入通道+输出通道+其他参数

```
if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, Low_IFM, SimConv):
```

### （5）Low_LAF

```
class low_LAF(nn.Module):
    def __init__(self, in_channels, out_channels):
    def forward(self, x):
```

只有输入通道+输出通道

没有对应条件的 咱就自己加

加在最后一个else上面

```
elif m is Low_LAF:
    c1, c2 = ch[f[1]], args[0]
    if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2]
```

### （6）Inject

```
class Inject(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_index: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
            global_inp=None,
    )
    def forward(self, x_l, x_g):
```

输入通道+输出通道+其他参数 

但是输入通道数有两个 因为前面跟的是Split模块

所以这里也需要单独处理

```
elif m is Inject:
	global_index = args[1]
	c1, c2 = ch[f[1]][global_index], args[0]
	if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
```

然后这里forward前向传播中有两个参数 咱之前接收的都只有一个参数 所以这得处理下

咱得记录下需要多参数的模块

````
m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
````

在这句代码下加入

```
# input_nums>1 说明有多个输入
if m is Inject:
    m.input_nums = len(f)
else:
    m.input_nums = 1
```

然后这里咱还需要修改predict_once函数 进行判断 如果input_nums > 1 咱需要通过解包传参

```
# x = m(x) 
# 将上注释改为
x = m(*x) if m.input_nums > 1 else m(x)
```

### （7）RepBlock

```
class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
    def forward(self, x):
```

输入通道+输出通道+其他参数 第一类

```
if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, Low_IFM, SimConv, RepBlock):
```

### （8）High_FAM

```
class High_FAM(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
    def forward(self, inputs):
```

只有参数，不需要传入输入输出通道数 输出通道数为输入通道数之和 

```
elif m in [Concat, Low_FAM, High_FAM]:
     # 没有参数
    c2 = sum(ch[x] for x in f)
```

### （9）High_IFM

```
class High_IFM(nn.Module):
    def __init__(self, block_num, embed_dim_n, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path_rate=0.1,
                 depths=2, norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
    def forward(self, x):
```

不需要输入通道 也不需要输出通道 输出通道数变为embed_dim_n

单独编写

```
elif m is High_IFM:
	# 输出通道数为第二个参数
    c2 = args[1]
```

### （10）High_LAF

```
class High_LAF(nn.Module):
	def forward(self, x1, x2):
```

没有\__init__

也就是说 不需要输入参数 输出通道数 等于 输入通道数 之和

```
elif m is m in [Concat, Low_FAM, High_FAM, High_LAF]:
    # 没有参数
    c2 = sum(ch[x] for x in f)
```



但是前向传播参数有两个 因为前面咱已经设置过 这里就没有必要重新设置了

# 3. 编写配置文件yolov8n_gold_yolo.yaml文件

在yolov8.yaml文件(\ultralytics\cfg\models\v8)的基础上修改

基本模板（把YOLOv8 neck部分给去掉了）

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
# 这个通道数按照你的通道数修改
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
	
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

然后咱根据GOLD-YOLO forward部分源码来配置

```
(c2, c3, c4, c5) = input      
# Low-GD
## use conv fusion global info
low_align_feat = self.low_FAM(input)
```

对应的模块定义为

```
class Low_FAM(nn.Module):
    def __init__(self):
    def forward(self, x):
    	x_l, x_m, x_s, x_n = x
```

对应的配置文件为

```
- [[2, 4, 6, -1], 1, Low_FAM, []]  # 10  
```

源码部分为

```
self.low_IFM = nn.Sequential(
                Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(extra_cfg.embed_dim_p, extra_cfg.embed_dim_p) for _ in 		range(extra_cfg.fuse_block_num)],
                Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
low_fuse_feat = self.low_IFM(low_align_feat)
```

对应的模块定义为

```
class Low_IFM(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim_p, fuse_block_num):
```

对应的配置文件为

```
- [-1, 1, Low_IFM, [768, 96, 3]]  # 11
```

源码部分为
```
low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
```
模块定义部分为
```
class Split(nn.Module):
    def __init__(self, trans_channels):
```

配置文件为
```
- [-1, 1, Split, [512, 256]]  # 12
```

源码部分为
```
c5_half = self.reduce_layer_c5(c5)
```
模块定义部分为
```
class SimConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
```

配置文件为
```
- [9, 1, SimConv, [512, 1, 1]]  # 13
```

源码部分为
```
self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[3]],  # 512, 256
                out_channels=channels_list[5],  # 256
        )
p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
```
模块定义部分为
```
class Low_LAF(nn.Module):
    def __init__(self, in_channels, out_channels):
```

配置文件为
```
- [[4, 6, -1], 1, Low_LAF, [256]]  # 14
```

源码部分为
```
p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[5], channels_list[5], norm_cfg=extra_cfg.norm_cfg,activations=nn.ReLU6)
```
模块定义部分为
```
class Inject(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_index: int,
            norm_cfg_type='BN'
    ) -> None:
    def forward(self, x_l, x_g):
```

配置文件为
```
- [[-1, 12], 1, Inject, [512, 0, 'SyncBN']]  # 15
```

源码部分为
```
self.Rep_p4 = RepBlock(
                in_channels=channels_list[5], 
                out_channels=channels_list[5], 
                n=num_repeats[5],
                block=block
        )
p4 = self.Rep_p4(p4)
```
模块定义部分为
```
class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
```

配置文件为
```
- [-1, 1, RepBlock, [512, 4]]  # 16
```

源码部分为
```
## inject low-level global info to p3
p4_half = self.reduce_layer_p4(p4)
p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
p3 = self.Rep_p3(p3)
```
配置文件为
```
- [-1, 1, SimConv, [256, 1, 1]]  # 17
- [[2, 4, -1], 1, Low_LAF, [128]]  # 18
- [[-1, 12], 1, Inject, [256, 1, 'SyncBN']]  # 19
- [-1, 1, RepBlock, [256, 4]]  # 20
```

源码部分为
```
high_align_feat = self.high_FAM([p3, p4, c5])
self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
```
模块定义部分为
```
class High_FAM(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
    def forward(self, inputs):
    	out = [self.pool(inp, output_size) for inp in inputs]
```

配置文件为
```
- [[-1, 16, 9], 1, High_FAM, [2, 'torch']]  # 21
```

源码部分为
```
dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
self.high_IFM = TopBasicLayer(
        block_num=extra_cfg.depths, #  2
        embedding_dim=extra_cfg.embed_dim_n, # 352
        key_dim=extra_cfg.key_dim, # 8
        num_heads=extra_cfg.num_heads, # 4
        mlp_ratio=extra_cfg.mlp_ratios, # 1
        attn_ratio=extra_cfg.attn_ratios, # 2
        drop=0, attn_drop=0,
        drop_path=dpr,
        norm_cfg=extra_cfg.norm_cfg
)
high_fuse_feat = self.high_IFM(high_align_feat)
```
模块定义部分为
```
class High_IFM(nn.Module):
    def __init__(self, block_num, embed_dim_n, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path_rate=0.1,
                 depths=2, norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
```

配置文件为
```
- [-1, 1, High_IFM, [2, 448, 8, 4, 1, 2, 0, 0, 0.1, 2, 'SyncBN']]  # 22
```

源码部分为
```
high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
self.conv_1x1_n = nn.Conv2d(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), 1, 1, 0)
```
配置文件为
```
- [-1, 1, nn.Conv2d, [1536, 1, 1, 0]]  # 23
```

源码部分为
```
high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
```
模块定义部分为
```
class Split(nn.Module):
    def __init__(self, trans_channels):
```

配置文件为
```
- [-1, 1, Split, [512, 1024]]  # 24
```

源码部分为
```
n4_adjacent_info = self.LAF_n4(p3, p4_half)
self.LAF_n4 = AdvPoolFusion()
```
模块定义部分为
```
class High_LAF(nn.Module):
    def forward(self, x1, x2):
```

配置文件为
```
- [[20, 17], 1, High_LAF, []]  # 25
```

源码部分为
```
 n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
 self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[8], channels_list[8],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
```
模块定义部分为

```
# Inject
class Inject(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_index: int,
            norm_cfg_type='BN'
    ) -> None:
    def forward(self, x_l, x_g):
```

配置文件为
```
- [[-1, 24], 1, Inject, [512, 0, 'SyncBN']]  # 26
```

源码部分为
```
self.Rep_n4 = RepBlock(
                in_channels=channels_list[6] + channels_list[7],  # 128 + 128
                out_channels=channels_list[8],  # 256
                n=num_repeats[7],
                block=block
        )
n4 = self.Rep_n4(n4)
```
模块定义部分为
```
class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
```

配置文件为
```
- [-1, 1, RepBlock, [512, 4]]  # 27 n4
```

源码部分为

```
n5_adjacent_info = self.LAF_n5(n4, c5_half)
n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
n5 = self.Rep_n5(n5)
```
配置文件为
```
- [[-1, 13], 1, High_LAF, []]  # 28
- [[-1, 24], 1, Inject, [1024, 1, 'SyncBN']]  # 29
- [-1, 1, RepBlock, [1024, 4]]  # 30 n5
```

源码部分为
```
outputs = [p3, n4, n5]
```
配置文件为
```
- [[20, 27, 30], 1, Detect, [nc]]  # 31 Detect(P3, P4, P5)
```

## 最终配置文件为

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 2048]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
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
  # Low-GD
  ## use conv fusion global info
  - [[-1, 6, 4, 2], 1, Low_FAM, []]  # 10
  - [-1, 1, Low_IFM, [768, 96, 3]]    # 11
  - [-1, 1, Split, [512, 256]]  # 12

  ## inject low-level global info to p4
  - [9, 1, SimConv, [512, 1, 1]]  # 13 c5_half
  - [[4, 6, -1], 1, Low_LAF, [256]]  # 14
  - [[-1, 12], 1, Inject, [512, 0, 'SyncBN']]  # 15
  - [-1, 1, RepBlock, [512, 4]]  # 16 p4

  ## inject low-level global info to p3
  - [ -1, 1, SimConv, [ 256, 1, 1 ] ]  # 17 p4_half
  - [ [ 2, 4, -1 ], 1, Low_LAF, [ 128 ] ]  # 18
  - [ [ -1, 12 ], 1, Inject, [ 256, 1, 'SyncBN' ] ]  # 19
  - [ -1, 1, RepBlock, [256, 4] ]  # 20 p3

  # High-GD
  ## use transformer fusion global info
  - [[-1, 16, 9], 1, High_FAM, [2, 'torch']]  # 21
  - [-1, 1, High_IFM, [2, 448, 8, 4, 1, 2, 0, 0, 0.1, 2, 'SyncBN']]  # 22
  - [-1, 1, nn.Conv2d, [1536, 1, 1, 0]]  # 23
  - [-1, 1, Split, [512, 1024]]  # 24

  ## inject low-level global info to n4
  - [[20, 17], 1, High_LAF, []]  # 25
  - [[-1, 24], 1, Inject, [512, 0, 'SyncBN']]  # 26
  - [-1, 1, RepBlock, [512, 4]]  # 27 n4

  ## inject low-level global info to n5
  - [ [ -1, 13 ], 1, High_LAF, [ ] ]  # 28
  - [ [ -1, 24 ], 1, Inject, [ 1024, 1, 'SyncBN' ] ]  # 29
  - [ -1, 1, RepBlock, [ 1024, 4 ] ]  # 30 n5

  - [[20, 27, 30], 1, Detect, [nc]]  # 31 Detect(P3, P4, P5)

```

# 4. 运行配置文件找错

```
from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8n_gold_yolo.yaml')  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data='coco128.yaml', epochs=20, imgsz=640)
```

## ModuleNotFoundError: No module named 'mmcv'

```
pip install -U openmim
mim install mmcv
```

## NameError: name 'RepVGGBlock' is not defined

RepVGGBlock放在RepBlock前面定义（这可能涉及python语法，引用类必须先定义，而引用方法可以后定义）

## return F.conv2d(input, weight, bias, self.stride,RuntimeError: non-positive stride is not supported

nn.Conv2d没有输入通道

在tasks.py中第一类中加入nn.Conv2d

##  UserWarning: upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'

参数设置deterministic=False

## AMP: checks skipped . Unable to load YOLOv8n due to possible Ultralytics package modifications. Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False.

\ultralytics\engine\trainer.py中的文件

```
a = m(im, device=device, verbose=False)[0].boxes.data  # FP32 inference
```

tasks.py中改动

```
try:
    x = m(*x) if m.input_nums > 1 else m(x)
except AttributeError:
    x = m(x)
```

## module 'backend_interagg' has no attribute 'FigureCanvas'

\ultralytics\utils\plotting.py 导库中加入

```
import matplotlib
matplotlib.use('TkAgg')
```











