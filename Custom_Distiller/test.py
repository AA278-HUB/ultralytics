# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO
from ultralytics.models.yolo.detect.KDDetectionTrainer import KDDetectionTrainer

student_model =YOLO("yolo11n.yaml")
teacher_model=YOLO(r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Train3\best.pt")


student_model.trainer=KDDetectionTrainer(distiller='mgd',teacher=teacher_model)


# 配置训练参数（如数据集路径、批大小、学习率等）
student_model.trainer.train(
    data='coco8.yaml',  # 你的数据配置文件路径
    epochs=50,  # 训练的 epoch 数量
    batch_size=16,  # 每个批次的大小
    imgsz=640,  # 输入图像大小
    lr0=0.01,  # 初始学习率
    lrf=0.1,  # 最终学习率比例
    save=True,  # 是否保存模型
    device=-1,  # 使用的设备（可以是多个 GPU）
    name=f"KD_training_{datetime.now().strftime('%Y%m%d_%H%M')}"  # 保存模型的名称
)