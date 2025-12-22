# 获取当前脚本所在目录
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录（即桌面/ultralytics）到Python搜索路径
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)


# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO
# 模型配置文件

model_yaml_path = r"C:\Users\Hunger\Desktop\ultralytics\runs\detect\MobileNetV4_20251201_11_31\weights\last.pt"
# model_yaml_path = r"runs/detect/MobileNetV4_20251201_11_31/weights/epoch140.pt"
# data="coco8.yaml"
data = "Custom_dataset_cfg/vehicle_orientation.yaml"
# 预训练模型
if __name__ == '__main__':
    # 加载预训练模型
    # model =YOLO(model_yaml_path)
    model = YOLO(model_yaml_path)
    results=model.train(resume=True)

    # results = model.train(data=data,
    #                       epochs=200,
    #                       batch=16,
    #                       imgsz=640,
    #                       cos_lr=True,
    #                       close_mosaic=50,
    #                       save=True,
    #                       save_period=10,
    #                       device="0",
    #                       name="MobileNetV4"+"_"+datetime.now().strftime("%Y%m%d_%H_%M"))