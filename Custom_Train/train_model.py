# coding:utf-8      `
from ultralytics import YOLO

# 模型配置文件
model_yaml_path = "Custom_Model_cfg/MobileNetV4.yaml"
data = "coco8.yaml"
# 预训练模型
# pre_model_name = 'yolo11s-cls.pt'

if __name__ == '__main__':
    # 加载预训练模型
    # model =YOLO(model_yaml_path)
    model = YOLO(model_yaml_path)
    results = model.train(data="coco8.yaml",
                          epochs=100,
                          batch=8,
                          imgsz=640,
                          name="Custom Model")

