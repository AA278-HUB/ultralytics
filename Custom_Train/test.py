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
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev1_CBAM.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev1_CA.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev1_CA_1.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2_CA.yaml"
model_yaml_path = "Custom_Model_cfg/MobileNetV4.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2_ECA.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2_True.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2_CA_Detect.yaml"
# model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2_EMA.yaml"
# model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2_CA.yaml"
# model_yaml_path = "Custom_Model_cfg/yolo11_Detect_CA.yaml"
# model_yaml_path="Custom_Model_cfg/yolo11_Ghost_Rep_Bottleneck_shufflev2.yaml"
# model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11n.yaml"
model_yaml_path = r"Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2_CA_Detect_C2PSA_MLCA.yaml"
model_yaml_path="Custom_Model_cfg/yolo11_shufflev2.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep_shufflev2.yaml"
model_yaml_path ="Custom_Model_cfg/yolo11_Ghost_Rep2.yaml"
model_yaml_path ="Custom_Model_cfg/yolo11_Ghost_C2faster.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_1.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11n.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_2.yaml"

model_yaml_path = "Custom_Model_cfg_2/yolo11_Ghost_Rep_Dw_shufflev2_.yaml"
model_yaml_path = "Custom_Model_cfg_2/yolo11_Ghost_Rep_Ghost_shufflev2_.yaml"
model_yaml_path = "Custom_Model_cfg_2/yolo11_Ghost_Rep_Ghost_shufflev2_CA_BiFPN.yaml"
model_yaml_path = "Custom_Model_cfg_2/yolo11_Ghost_Rep_Ghost_shufflev2_CA_BiFPN_Fusion.yaml"
model_yaml_path = "Custom_Model_cfg_2/yolo11_Ghost_Rep_Ghost_shufflev2_CA_BiFPN_FusionV2.yaml"

model_yaml_path="Custom_Model_cfg_3/test.yaml"
# model_yaml_path="Custom_Model_cfg_2/yolo11_Ghost_Rep_Ghost_shufflev2_CA_SimAM.yaml"
model_yaml_path="Custom_Model_cfg_3/yolo11_RepViTBlock_Ghost_shufflev2.yaml"
# model_yaml_path = "Custom_Model_cfg_2/test.yaml"
model_yaml_path = r"Custom_Model_cfg_3/yolo11_RepViTBlock_Ghost.yaml"

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
