# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO

# 模型配置文件
model_yaml_path = "Custom_Model_cfg/MobileNetV4.yaml"
data = "Custom_dataset_cfg/coco-vehicle.yaml"
date_str = datetime.now().strftime("%Y%m%d_%H_%M")
print(date_str)
# 预训练模型
if __name__ == '__main__':
    # 加载预训练模型
    # model =YOLO(model_yaml_path)
    model = YOLO(model_yaml_path)
    results = model.train(data=data,
                          epochs=200,
                          batch=8,
                          imgsz=640,
                          cos_lr=True,
                          close_mosaic=50,
                          name="test"+str(date_str))

