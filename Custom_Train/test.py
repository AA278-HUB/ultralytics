# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO

# 模型配置文件
model_yaml_path = "Custom_Model_cfg/MobileNetV4.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_dyhead.yaml"

data="coco8.yaml"
data = "Custom_dataset_cfg/coco-vehicle.yaml"
# 预训练模型
if __name__ == '__main__':
    # 加载预训练模型
    # model =YOLO(model_yaml_path)
    model = YOLO(model_yaml_path)

    results = model.train(data=data,
                          epochs=10,
                          batch=8,
                          imgsz=640,
                          cos_lr=True,
                          close_mosaic=50,
                          name="test"+datetime.now().strftime("%Y%m%d_%H_%M"))

