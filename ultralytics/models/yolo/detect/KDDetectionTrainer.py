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
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping, TORCH_2_4


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
    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)

        # Load teacher model to device
        if self.teacher is not None:
            for k, v in self.teacher.named_parameters():
                v.requires_grad = True
            self.teacher = self.teacher.to(self.device)

        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

            if self.teacher is not None:
                self.teacher = nn.parallel.DistributedDataParallel(self.teacher, device_ids=[RANK])
                temp = self.teacher.eval()

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(
                model=self.model,
                imgsz=self.args.imgsz,
                amp=self.amp,
                batch=self.batch_size,
            )

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            teacher=self.teacher,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")
    def preprocess_batch(self, batch):
        # Standard preprocessing
        return super().preprocess_batch(batch)
    def _do_train(self, world_size=1):
        """
        重写_do_train方法以插入知识蒸馏逻辑。
        这是一个最小重写；大多数逻辑从BaseTrainer继承。
        """
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # 批次数量
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 预热迭代次数
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        epoch = self.start_epoch
        self.optimizer.zero_grad()  # 零化任何恢复的梯度以确保训练开始时的稳定性
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 抑制 'Detected lr_scheduler.step() before optimizer.step()' 警告
                self.scheduler.step()

            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # 更新数据加载器属性（可选）
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None

            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # 预热
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x 插值
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # 偏置学习率从0.1下降到lr0，其他学习率从0.0上升到lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # 前向传播
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)

                    # 教师前向（捕获特征）
                    with torch.no_grad():
                        _ = self.teacher.model(batch['img'])  # 运行推理以捕获特征
                        teacher_features = self.teacher_features

                    # 学生前向和损失（内部捕获特征）
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                    # 使用捕获的特征计算KD损失
                    student_features = self.student_features
                    kd_loss = self.feature_loss(student_features, teacher_features)

                    # 将KD损失添加到总损失中
                    self.loss += kd_loss

                    # 将kd_loss附加到loss_items以进行日志记录
                    self.loss_items = torch.cat((self.loss_items, kd_loss.detach().cpu().unsqueeze(0)))

                # 反向传播
                self.scaler.scale(self.loss).backward()

                # 优化 - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # 时间停止
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # 如果是DDP训练
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # 将'stop'广播到所有rank
                            self.stop = broadcast_list[0]
                        if self.stop:  # 训练时间超过
                            break

                # 日志
                if RANK in {-1, 0}:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                    losses_str = f"{'/'.join(f'{loss:.3g}' for loss in self.tloss)}" if loss_len > 1 else f'{self.tloss:.3g}'
                    pbar.set_description(
                        f'{epoch + 1}/{self.epochs} {mem} {losses_str}'
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 用于日志记录器
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # 验证
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # 保存模型
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # 调度器
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # 不要移动
                self.stop |= epoch >= self.epochs  # 如果超过epochs则停止
            self.run_callbacks("on_fit_epoch_end")
            torch.cuda.empty_cache()  # 在epoch结束时清除GPU内存，可能有助于减少CUDA内存不足错误

            # 早停
            if RANK != -1:  # 如果是DDP训练
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # 将'stop'广播到所有rank
                self.stop = broadcast_list[0]
            if self.stop:
                break  # 必须打破所有DDP rank
            epoch += 1

        if RANK in {-1, 0}:
            # 使用best.pt进行最终验证
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")