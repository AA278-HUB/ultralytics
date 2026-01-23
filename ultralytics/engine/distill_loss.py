import torch
from torch import nn


class MGDLoss(nn.Module):

    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):

        super(MGDLoss, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_mgd = alpha_mgd

        self.lambda_mgd = lambda_mgd

        self.generation = [

            nn.Sequential(

                nn.Conv2d(channel_s, channel, kernel_size=3, padding=1),

                nn.ReLU(inplace=True),

                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel_s, channel in

            zip(channels_s, channels_t)

        ]

    def forward(self, y_s, y_t, layer=None):

        """Forward computation.

        Args:

            y_s (list): The student model prediction with

                shape (N, C, H, W) in list.

            y_t (list): The teacher model prediction with

                shape (N, C, H, W) in list.

        Return:

            torch.Tensor: The calculated loss value of all stages.

        """

        assert len(y_s) == len(y_t)

        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):

            # print(s.shape)

            # print(t.shape)

            # assert s.shape == t.shape

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

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()

        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.

        Args:

            y_s (list): The student model prediction with

                shape (N, C, H, W) in list.

            y_t (list): The teacher model prediction with

                shape (N, C, H, W) in list.

        Return:

            torch.Tensor: The calculated loss value of all stages.

        """

        assert len(y_s) == len(y_t)

        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # normalize in channel diemension

            import torch.nn.functional as F

            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)

            cost = torch.sum(

                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -

                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))

        loss = sum(losses)

        return loss


class FeatureLoss(nn.Module):

    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):

        super(FeatureLoss, self).__init__()

        self.loss_weight = loss_weight

        self.distiller = distiller

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.align_module = nn.ModuleList([

            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1, padding=0).to(device)

            for channel, tea_channel in zip(channels_s, channels_t)

        ])

        self.norm = [

            nn.BatchNorm2d(tea_channel, affine=False).to(device)

            for tea_channel in channels_t

        ]

        self.norm1 = [

            nn.BatchNorm2d(set_channel, affine=False).to(device)

            for set_channel in channels_s

        ]

        if distiller == 'mgd':

            self.feature_loss = MGDLoss(channels_s, channels_t)

        elif distiller == 'cwd':

            self.feature_loss = CWDLoss(channels_s, channels_t)

        else:

            raise NotImplementedError

    def forward(self, y_s, y_t):

        assert len(y_s) == len(y_t)

        tea_feats = []

        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):

            if self.distiller == 'cwd':

                s = self.align_module[idx](s)

                s = self.norm[idx](s)

            else:

                s = self.norm1[idx](s)

            t = self.norm[idx](t)

            tea_feats.append(t)

            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)

        return self.loss_weight * loss

class DistillationLoss:
    """
    特征蒸馏损失类，用于在教师模型 (modelL) 和学生模型 (modeln) 之间进行特征对齐。
    支持指定层（YOLO 结构中的特定 Detect 层 cv2 输出），通过 forward hook 捕获特征图，
    然后使用 FeatureLoss 计算蒸馏损失。

    假设已有一个 FeatureLoss 类，能够接受通道列表并处理多层特征图列表。
    """

    def __init__(self, model_teacher, model_student, distiller="cwd"):
        """
        初始化蒸馏损失。

        Args:
            model_teacher: 教师模型（已去掉 DataParallel 包装）
            model_student: 学生模型（已去掉 DataParallel 包装）
            distiller: 蒸馏损失类型（如 "CWDLoss"），前三个字符用于 FeatureLoss
        """
        self.distiller = distiller

        # 指定需要蒸馏的层
        layers = ["6", "8", "12", "15", "18", "21"]  # 可根据需要调整

        # 收集教师和学生对应层的输出通道数
        channels_t = self._get_layer_channels(model_teacher, layers)
        channels_s = self._get_layer_channels(model_student, layers)

        # 创建特征蒸馏损失函数
        self.D_loss_fn = FeatureLoss(
            channels_s=channels_s,
            channels_t=channels_t,
            distiller=distiller
        )

        # 收集需要 hook 的模块（cv2.conv）
        self.teacher_modules = self._get_target_modules(model_teacher, layers)
        self.student_modules = self._get_target_modules(model_student, layers)

        # 存储 hook 句柄，用于后续移除
        self.hooks = []

    def _get_layer_channels(self, model, layers):
        """提取指定 layers 中 cv2.conv 的输出通道数"""
        channels = []
        for name, module in model.named_modules():
            parts = name.split(".")
            # 处理 DataParallel 包装的 "module." 前缀
            if parts and parts[0] == "module":
                parts = parts[1:]
            if len(parts) == 3 and parts[1] in layers and "cv2" in parts[2]:
                channels.append(module.conv.out_channels)
        # 取最后 len(layers) 个，确保顺序对应
        return channels[-len(layers):]

    def _get_target_modules(self, model, layers):
        """收集指定 layers 中 cv2 模块（用于注册 hook）"""
        modules = []
        for name, module in model.named_modules():
            parts = name.split(".")
            if parts and parts[0] == "module":
                parts = parts[1:]
            if len(parts) == 3 and parts[1] in layers and "cv2" in name:
                modules.append(module)
        return modules

    def register_hooks(self):
        """注册 forward hook，捕获教师和学生对应层的特征图"""
        self.teacher_features = []
        self.student_features = []

        def make_hook(feature_list):
            def hook(module, input, output):
                feature_list.append(output)

            return hook

        for t_mod, s_mod in zip(self.teacher_modules, self.student_modules):
            self.hooks.append(t_mod.register_forward_hook(make_hook(self.teacher_features)))
            self.hooks.append(s_mod.register_forward_hook(make_hook(self.student_features)))

    def get_loss(self):
        """计算蒸馏损失并清空缓存的特征"""
        if not self.teacher_features or not self.student_features:
            return 0.0

        distill_loss = self.D_loss_fn(
            y_t=self.teacher_features,
            y_s=self.student_features
        )

        # 非 CWD 损失时降低权重（可根据实验调整）
        if self.distiller.lower() != 'cwd':
            distill_loss *= 0.3

        # 清空特征缓存，准备下一轮 forward
        self.teacher_features.clear()
        self.student_features.clear()

        return distill_loss

    def remove_hooks(self):
        """移除所有注册的 hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class DetectInputDistillationLoss:
    """
    特征蒸馏损失类（Detect输入版），用于在教师模型和学生模型之间对齐Detect模块的输入特征（即P3/P4/P5三尺度特征图）。
    与原DistillationLoss不同，本版本直接hook Detect模块的forward_pre_hook，捕获传入Detect的特征列表[x0, x1, x2]（对应P3、P4、P5）。
    这是一种更简洁的方式，只蒸馏最终送入Detect的三层特征，无需依赖中间层的命名规则（如cv2）。

    兼容原使用方式（方法名不变、数量不少），便于直接迁移替换。
    假设已有一个FeatureLoss类，能够接受通道列表并处理多层特征图列表（这里固定为3层）。
    """

    def __init__(self, model_teacher, model_student, distiller="cwd"):
        """
        初始化蒸馏损失。

        Args:
            model_teacher: 教师模型（已去掉DataParallel包装，需为YOLO(...).model）
            model_student: 学生模型（已去掉DataParallel包装，需为YOLO(...).model）
            distiller: 蒸馏损失类型（如 "CWDLoss"），前三个字符用于FeatureLoss
        """
        self.distiller = distiller

        # 直接取Detect模块（model[-1]）
        self.teacher_detect = model_teacher.model[-1]
        self.student_detect = model_student.model[-1]

        # 从Detect模块的.ch属性自动获取输入通道数（YOLOv8/v11的Detect都有self.ch = (ch0, ch1, ch2)）
        channels_t = self._get_detect_input_channels(self.teacher_detect)
        channels_s = self._get_detect_input_channels(self.student_detect)

        # 创建特征蒸馏损失函数（固定3层）
        self.D_loss_fn = FeatureLoss(
            channels_s=channels_s,
            channels_t=channels_t,
            distiller=distiller[:3]
        )

        # 存储hook句柄，用于后续移除
        self.hooks = []

        # 下面的两个私有方法保留（兼容原类结构），但在新逻辑中不再使用（返回空或固定值均可）
        # 如果你仍想保留原多层逻辑，可注释掉直接hook部分，改回调用它们
        self._dummy_layers = None  # 占位

    def _get_layer_channels(self, model, layers=None):
        """占位方法（兼容原类），在新版本中自动从Detect.ch获取，无需调用"""
        return list(model.model[-1].ch)

    def _get_target_modules(self, model, layers=None):
        """占位方法（兼容原类），在新版本中直接使用Detect模块"""
        return [model.model[-1]]

    def register_hooks(self):
        """注册forward_pre_hook，直接捕获传入Detect的特征列表[P3, P4, P5]"""
        self.teacher_features = []
        self.student_features = []

        def make_pre_hook(feature_list):
            def hook(module, input):
                # input是一个tuple，input[0]就是传入Detect的特征列表[P3, P4, P5]
                feats = input[0]
                # 建议detach避免梯度/内存问题
                feature_list.append([f.detach() for f in feats])
            return hook

        # 为教师和学生Detect注册pre_hook
        self.hooks.append(self.teacher_detect.register_forward_pre_hook(
            make_pre_hook(self.teacher_features)))
        self.hooks.append(self.student_detect.register_forward_pre_hook(
            make_pre_hook(self.student_features)))

    def get_loss(self):
        """计算蒸馏损失并清空缓存的特征"""
        if not self.teacher_features or not self.student_features:
            return 0.0

        # 每次forward只会append一次（一个batch），取[0]即可得到[P3, P4, P5]列表
        y_t = self.teacher_features[0]   # list of 3 tensors
        y_s = self.student_features[0]   # list of 3 tensors

        distill_loss = self.D_loss_fn(
            y_t=y_t,
            y_s=y_s
        )

        # 非CWD损失时降低权重（与原版保持一致，可根据实验调整）
        if self.distiller.lower() != 'cwd':
            distill_loss *= 0.3

        # 清空缓存，准备下一轮forward
        self.teacher_features.clear()
        self.student_features.clear()

        return distill_loss

    def remove_hooks(self):
        """移除所有注册的hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    def _get_detect_input_channels(self, detect_module):
        """从Detect.cv2推断输入通道数（兼容YOLOv11等无.ch的版本）"""
        # Detect.cv2 是 ModuleList，长度为3，每个元素是Sequential，第一个模块是Conv（处理输入通道）
        # 示例：detect_module.cv2[0][0].conv.in_channels → P3输入通道
        #        detect_module.cv2[1][0].conv.in_channels → P4输入通道
        #        detect_module.cv2[2][0].conv.in_channels → P5输入通道
        channels = []
        for branch in detect_module.cv2:
            first_conv = branch[0].conv  # 第一个Conv模块（Conv2d）
            channels.append(first_conv.in_channels)
        return channels


