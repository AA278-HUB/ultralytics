# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO

# 模型配置文件
model_yaml_path = "Custom_Model_cfg/yolo11_dyhead.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_1.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep.yaml"
data="coco8.yaml"
#data = "Custom_dataset_cfg/coco-vehicle.yaml"
# 预训练模型
if __name__ == '__main__':



    model = YOLO(model_yaml_path)
    model.info(detailed=True)
    results = model.train(data=data,
                          epochs=3,
                          batch=8,
                          imgsz=640,
                          cos_lr=True,
                          close_mosaic=50,
                          save= True,
                          device="0",
                          workers=16,
                          name="test"+datetime.now().strftime("%Y%m%d_%H_%M"))
    # 部署准备
    model.eval()
    for m in model.modules():
        if hasattr(m, 'fuse_convs'):
            m.fuse_convs()
    model.info(detailed=True)
    model.val(data=data,plots=True)
