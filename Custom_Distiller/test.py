from datetime import datetime

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer




if __name__ == '__main__':
    # -------------------------
    # 1. 加载模型
    # -------------------------
    student = YOLO("yolo11n.yaml")
    teacher = YOLO(r"C:\Users\Hunger\Desktop\ultralytics\Custom_Distiller\best.pt")  # 必须同尺度
    teacher.model.eval()
    for p in teacher.model.parameters():
        p.requires_grad = False

    # -------------------------
    # 2. 特征缓存
    # -------------------------
    s_feats, t_feats = None, None

    def save_student_feats(m, inp):
        global s_feats
        s_feats = inp[0]   # [P3, P4, P5]

    def save_teacher_feats(m, inp):
        global t_feats
        t_feats = inp[0]

    # Detect 是 model.model[-1]
    student.model.model[-1].register_forward_pre_hook(save_student_feats)
    teacher.model.model[-1].register_forward_pre_hook(save_teacher_feats)

    # -------------------------
    # 3. override loss
    # -------------------------
    kd_weight = 1.0
    original_loss = student.model.loss

    def kd_loss(self, batch, preds=None):
        global s_feats, t_feats

        # 确保 criterion 已初始化
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        # 原 YOLO loss（这里会触发 student hook）
        if preds is None:
            preds = self.forward(batch["img"])
        total_loss, loss_items = original_loss(batch, preds)

        # teacher forward（只为抓特征）
        with torch.no_grad():
            _ = teacher.model(batch["img"])

        # KD loss（无 adapter，直接 MSE）
        kd_term = 0.0
        for fs, ft in zip(s_feats, t_feats):
            kd_term += F.mse_loss(fs, ft.detach())

        total_loss = total_loss + kd_weight * kd_term

        # 日志里加一项 KD
        loss_items = torch.cat([
            loss_items,
            kd_term.detach().unsqueeze(0)
        ])

        s_feats, t_feats = None, None
        return total_loss, loss_items


# 正确绑定 self
    student.loss = kd_loss.__get__(student.model, type(student.model))
    data = "Custom_dataset_cfg/vehicle_orientation_mini.yaml"
    model_name="Distill_Model"
    student.train(
                data=data,
                epochs=200,
                batch=32,
                imgsz=640,  # 保持不变
                cos_lr=True,
                lr0=0.01,  # ↑ 初始学习率
                lrf=0.1,  # ↑ 最终学习率比例
                close_mosaic=20,  # 提前关闭 mosaic
                save=True,
                device=0,
                name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H_%M')}"
            )





































# # coding:utf-8      `
# from datetime import datetime
# from ultralytics import YOLO
# from ultralytics.models.yolo.detect import DetectionTrainer
# from ultralytics.models.yolo.detect.KDDetectionTrainer import KDDetectionTrainer
# from ultralytics.utils import DEFAULT_CFG
#
# student_model =YOLO("yolo11n.yaml")
# teacher_model=YOLO(r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Distiller\best.pt")
#
# args = dict(model=student_model,data="coco8.yaml", epochs=3)
# trainer = KDDetectionTrainer(cfg=DEFAULT_CFG,student=student_model, teacher=teacher_model,overrides=args)
#
# trainer.train()
#
#
#
#
# # trainer =DetectionTrainer(cfg=DEFAULT_CFG,overrides=args)
# # student_model.trainer=KDDetectionTrainer(distiller='mgd',teacher=teacher_model)
# # 配置训练参数（如数据集路径、批大小、学习率等）