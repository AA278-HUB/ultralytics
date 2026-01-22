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
    teacher = YOLO(r"C:\Users\Hunger\Desktop\ultralytics\Custom_Distiller\best.pt")  # 必须同尺度
    student.train(epochs=3,data="Custom_dataset_cfg/vehicle_orientation_mini.yaml",Distill=True,Teacher=teacher.model,distill_loss="cwd")