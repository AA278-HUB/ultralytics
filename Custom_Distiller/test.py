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
    student = YOLO(r"DistillModel/yolo11n.yaml")
    teacher = YOLO(r"C:\Users\Hunger\Desktop\ultralytics\Custom_Distiller\best_m.pt")  # 必须同尺度（imgsz相同）

    # # 教师模型冻结 + eval
    # teacher.model.eval()
    # for p in teacher.model.parameters():
    #     p.requires_grad = False

    # -------------------------
    # 2. 知识蒸馏训练参数（结合你以前的常规训练参数）
    # -------------------------
    student.train(
        data="Custom_dataset_cfg/vehicle_orientation_mini.yaml",   # 你的数据集配置文件
        epochs=200,                            # 完整训练轮数
        batch=32,                              # batch size
        imgsz=640,                             # 输入图像尺寸（教师和学生必须保持一致）
        cos_lr=True,                           # 使用余弦学习率调度
        lr0=0.01,                              # 初始学习率
        lrf=0.1,                               # 最终学习率比例（cos_lr 下有效）
        close_mosaic=20,                       # 最后 20 个 epoch 关闭 mosaic 数据增强
        save=True,                             # 保存最佳和最新权重
        device=0,                             # 你的自定义设备选择（-1 可能代表自动或 CPU）
        # amp=False,                             # 关闭混合精度（如果你显存充足或想更稳定可设 True）
        name=f"yolo11n_distill_{datetime.now().strftime('%Y%m%d_%H_%M')}",  # 实验文件夹名，带时间戳
        # --------------------- 知识蒸馏专属参数 ---------------------
        Distill=True,                          # 开启蒸馏模式（你的自定义开关）
        Teacher=teacher.model,                 # 传入教师模型（已冻结）
        distill_loss="cwd"                     # 蒸馏损失类型（cwd / mgd 等）
    )