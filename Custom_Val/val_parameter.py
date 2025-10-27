# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO
# 模型配置文件
model_yaml_path = "Custom_Model_cfg/yolo11_dyhead.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_1.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep.yaml"
data="coco8.yaml"
model = YOLO(model_yaml_path)
model.info()
# 部署准备
model.fuse()
model.eval()
model.val()
