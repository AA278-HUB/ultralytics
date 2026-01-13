# # --------------------------------------------------------
# # Deformable Convolution v4
# # Copyright (c) 2023 OpenGVLab
# # Licensed under The MIT License [see LICENSE for details]
# # --------------------------------------------------------
#
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division
#
# import math
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.nn.init import xavier_uniform_, constant_
# from ultralytics.nn.modules import Conv
# from ultralytics.nn.modules.DCNv4.Common_DCNv4 import DCNv4Function
#
#
# class CenterFeatureScaleModule(nn.Module):
#     def forward(self,
#                 query,
#                 center_feature_scale_proj_weight,
#                 center_feature_scale_proj_bias):
#         center_feature_scale = F.linear(query,
#                                         weight=center_feature_scale_proj_weight,
#                                         bias=center_feature_scale_proj_bias).sigmoid()
#         return center_feature_scale
#
# class DCNv4(nn.Module):
#     def __init__(
#             self,
#             channels=64,
#             kernel_size=3,
#             stride=1,
#             pad=1,
#             dilation=1,
#             group=4,
#             offset_scale=1.0,
#             dw_kernel_size=None,
#             center_feature_scale=False,
#             remove_center=False,
#             output_bias=True,
#             without_pointwise=False,
#             **kwargs):
#         """
#         DCNv4 Module
#         :param channels
#         :param kernel_size
#         :param stride
#         :param pad
#         :param dilation
#         :param group
#         :param offset_scale
#         :param act_layer
#         :param norm_layer
#         """
#         super().__init__()
#         if channels % group != 0:
#             raise ValueError(
#                 f'channels must be divisible by group, but got {channels} and {group}')
#         _d_per_group = channels // group
#
#         # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
#         assert _d_per_group % 16 == 0
#
#         self.offset_scale = offset_scale
#         self.channels = channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
#         self.pad = pad
#         self.group = group
#         self.group_channels = channels // group
#         self.offset_scale = offset_scale
#         self.dw_kernel_size = dw_kernel_size
#         self.center_feature_scale = center_feature_scale
#         self.remove_center = int(remove_center)
#         self.without_pointwise = without_pointwise
#
#         self.K =  group * (kernel_size * kernel_size - self.remove_center)
#         if dw_kernel_size is not None:
#             self.offset_mask_dw = nn.Conv2d(channels, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
#         self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3)/8)*8))
#         if not without_pointwise:
#             self.value_proj = nn.Linear(channels, channels)
#             self.output_proj = nn.Linear(channels, channels, bias=output_bias)
#         self._reset_parameters()
#
#         if center_feature_scale:
#             self.center_feature_scale_proj_weight = nn.Parameter(
#                 torch.zeros((group, channels), dtype=torch.float))
#             self.center_feature_scale_proj_bias = nn.Parameter(
#                 torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
#             self.center_feature_scale_module = CenterFeatureScaleModule()
#
#     def _reset_parameters(self):
#         constant_(self.offset_mask.weight.data, 0.)
#         constant_(self.offset_mask.bias.data, 0.)
#         if not self.without_pointwise:
#             xavier_uniform_(self.value_proj.weight.data)
#             constant_(self.value_proj.bias.data, 0.)
#             xavier_uniform_(self.output_proj.weight.data)
#             if self.output_proj.bias is not None:
#                 constant_(self.output_proj.bias.data, 0.)
#
#     def forward(self, input, shape=None):
#         """
#         :param query                       (N, H, W, C)
#         :return output                     (N, H, W, C)
#         """
#         N, L, C = input.shape
#         if shape is not None:
#             H, W = shape
#         else:
#             H, W = int(L**0.5), int(L**0.5)
#
#
#         x = input
#         if not self.without_pointwise:
#             x = self.value_proj(x)
#         x = x.reshape(N, H, W, -1)
#         if self.dw_kernel_size is not None:
#             offset_mask_input = self.offset_mask_dw(input.view(N, H, W, C).permute(0, 3, 1, 2))
#             offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
#         else:
#             offset_mask_input = input
#         offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)
#
#         x_proj = x
#
#         x = DCNv4Function.apply(
#             x, offset_mask,
#             self.kernel_size, self.kernel_size,
#             self.stride, self.stride,
#             self.pad, self.pad,
#             self.dilation, self.dilation,
#             self.group, self.group_channels,
#             self.offset_scale,
#             256,
#             self.remove_center
#             )
#
#         if self.center_feature_scale:
#             center_feature_scale = self.center_feature_scale_module(
#                 x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
#             center_feature_scale = center_feature_scale[..., None].repeat(
#                 1, 1, 1, 1, self.channels // self.group).flatten(-2)
#             x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
#
#         x = x.view(N, L, -1)
#
#         if not self.without_pointwise:
#             x = self.output_proj(x)
#         return x
# class DCNv4Conv2d(nn.Module):
#     """
#     DCNv4 wrapper for NCHW feature maps
#     """
#
#     def __init__(
#         self,
#         channels,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         dilation=1,
#         group=4,
#         offset_scale=1.0,
#         **kwargs
#     ):
#         super().__init__()
#
#         self.channels = channels
#         self.stride = stride
#
#         self.dcnv4 = DCNv4(
#             channels=channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             pad=padding,
#             dilation=dilation,
#             group=group,
#             offset_scale=offset_scale,
#             **kwargs
#         )
#
#     def forward(self, x):
#         """
#         x: (N, C, H, W)
#         return: (N, C, H, W)
#         """
#         n, c, h, w = x.shape
#
#         # flatten -> (N, L, C)
#         x = x.permute(0, 2, 3, 1).contiguous().view(n, h * w, c)
#
#         x = self.dcnv4(x, shape=(h, w))
#
#         # restore -> (N, C, H, W)
#         x = x.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
#
#         return x
#
# class Bottleneck_DCNv4(nn.Module):
#     """YOLO Bottleneck with DCNv4"""
#
#     def __init__(
#         self,
#         c1,
#         c2,
#         shortcut=True,
#         g=1,
#         e=0.5,
#         dcn_group=4
#     ):
#         super().__init__()
#         c_ = int(c2 * e)
#
#         # 1x1 conv (普通卷积)
#         self.cv1 = Conv(c1, c_, 1, 1)
#
#         # 3x3 DCNv4
#         self.cv2 = DCNv4Conv2d(
#             channels=c_,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             group=dcn_group,
#         )
#
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         y = self.cv2(self.cv1(x))
#         return x + y if self.add else y
# class C2f_DCNv4(nn.Module):
#     """C2f with DCNv4 Bottleneck"""
#
#     def __init__(
#         self,
#         c1,
#         c2,
#         n=1,
#         shortcut=False,
#         g=1,
#         e=0.5,
#         dcn_group=4
#     ):
#         super().__init__()
#         self.c = int(c2 * e)
#
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)
#
#         self.m = nn.ModuleList(
#             Bottleneck_DCNv4(
#                 self.c,
#                 self.c,
#                 shortcut=shortcut,
#                 g=g,
#                 e=1.0,
#                 dcn_group=dcn_group
#             )
#             for _ in range(n)
#         )
#
#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
