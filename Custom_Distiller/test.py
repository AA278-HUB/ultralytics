from datetime import datetime

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModelWithKD
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer




if __name__ == '__main__':
    # -------------------------
    # 1. 加载模型
    # -------------------------
    student = YOLO(r"DistillModel/yolo11.yaml")
    teacher = YOLO(r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Distiller\yolo11n.pt")  # 必须同尺度
    teacher.model.eval()
    for p in teacher.model.parameters():
        p.requires_grad = False
    student.model=DetectionModelWithKD(cfg=r"DistillModel/yolo11.yaml",ch=3,nc=4,verbose=False,teacher=teacher,kd_weight=1)
    student.train(data="coco8.yaml",epochs=3,Distill=True,model=student)