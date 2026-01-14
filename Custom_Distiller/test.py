# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.KDDetectionTrainer import KDDetectionTrainer
from ultralytics.utils import DEFAULT_CFG

student_model =YOLO("yolo11n.yaml")
teacher_model=YOLO(r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Distiller\best.pt")

args = dict(model=student_model,data="coco8.yaml", epochs=3)
trainer = KDDetectionTrainer(cfg=DEFAULT_CFG,student=student_model, teacher=teacher_model,overrides=args)

trainer.train()




# trainer =DetectionTrainer(cfg=DEFAULT_CFG,overrides=args)
# student_model.trainer=KDDetectionTrainer(distiller='mgd',teacher=teacher_model)
# 配置训练参数（如数据集路径、批大小、学习率等）