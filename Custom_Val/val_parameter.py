# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO
# 模型配置文件
# 模型路径列表
model_path = [
    r"Custom_Model_cfg/yolo11_Ghost_1.yaml",
    r"Custom_Model_cfg/yolo11n.yaml",
    r"Custom_Model_cfg/yolo12n.yaml",
    r"Custom_Model_cfg/yolov5n.yaml",
    r"Custom_Model_cfg/yolov6n.yaml",
    r"Custom_Model_cfg/yolov8n.yaml",
    r"Custom_Model_cfg/yolov9t.yaml",
    r"Custom_Model_cfg/MobileNetV4.yaml",
    r"Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2.yaml",
    r"Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2_CA.yaml",
]


for path in model_path:

    model = YOLO(path)
    model.info()
    # results = model.train(data="coco8.yaml", epochs=1, imgsz=640)
    # 部署准备
    model.fuse()
    # model.info(True)
    # model.eval()
    # model.val()
