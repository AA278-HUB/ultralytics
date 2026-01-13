import math
import warnings
from datetime import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK, LOGGER, TQDM, colorstr, dist, LOCAL_RANK, callbacks
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_imgsz, check_amp
from ultralytics.utils.loss import KDDetectionLoss
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping, TORCH_2_4, unwrap_model, autocast, \
    unset_deterministic


# CWDLoss 类定义（通道-wise 蒸馏损失）
class CWDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # 在通道维度上归一化
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)
            ) * (self.tau ** 2)
            losses.append(cost / (C * N))
        loss = sum(losses)
        return loss

# MGDLoss 类定义（掩码生成蒸馏损失）
class MGDLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generation = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        ).to(device) for channel in teacher_channels])

    def forward(self, y_s, y_t, layer=None):
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)
        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss

# FeatureLoss 类定义（广义特征损失）
class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        for s_chan, t_chan in zip(channels_s, channels_t):
            align = nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(t_chan, affine=False)
            ).to(device)
            self.align_module.append(align)
        for t_chan in channels_t:
            self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
        for s_chan in channels_s:
            self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))
        if distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        if len(y_s) != len(y_t):
            y_t = y_t[len(y_t) // 2:]
        tea_feats = []
        stu_feats = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = s.type(next(self.align_module[idx].parameters()).dtype)
            t = t.type(next(self.align_module[idx].parameters()).dtype)
            if self.distiller == "cwd":
                s = self.align_module[idx](s)
                stu_feats.append(s)
                tea_feats.append(t.detach())
            else:
                t = self.norm1[idx](t)
                stu_feats.append(s)
                tea_feats.append(t.detach())
        loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss

# KDDetectionTrainer 类定义
class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, distiller='mgd', loss_weight=1.0, _callbacks=None,teacher=None,student=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        # 加载教师模型
        # self.teacher = YOLO('yolo11x.pt')  # 假设你有一个预训练的YOLOv11x模型
        self.model=student
        self.teacher=teacher
        self.teacher.model.eval()  # 将教师模型设置为评估模式
        for param in self.teacher.model.parameters():
            param.requires_grad = False  # 教师模型不需要梯度
        self.model.loss = KDDetectionLoss(
            model=self.model,
            teacher=self.teacher.model,
            kd_weight=loss_weight
        )

