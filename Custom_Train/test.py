# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO

# 模型配置文件
model_yaml_path = "Custom_Model_cfg/yolo11_dyhead.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_1.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shuffle.yaml"
# model_yaml_path = "Custom_Model_cfg/yolo11n.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shuffle_2.yaml"
# model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev1_CBAM.yaml"

data="coco8.yaml"
#data = "Custom_dataset_cfg/coco-vehicle.yaml"
# 预训练模型
if __name__ == '__main__':

    model = YOLO(model_yaml_path)
    results = model.train(data=data,
                          epochs=3,
                          batch=8,
                          imgsz=640,
                          cos_lr=True,
                          close_mosaic=50,
                          save= True,
                          device=-1,
                          workers=16,
                          name="test"+datetime.now().strftime("%Y%m%d_%H_%M"))

    print("训练完成后打印参数")

    model.info()
    # 部署准备
    model.fuse()
    model.eval()
    model.val()
    print("训练结束打印参数")
    model.info()
