# coding: utf-8
from datetime import datetime
from ultralytics import YOLO
import os

# 模型配置文件列表
model_yaml_paths = [
    # r"Custom_Model_cfg_3/yolo11_Ghost_Ghost_shufflev2.yaml",
    # r"Custom_Model_cfg_3/yolo11_Pconv_Ghost_shufflev2.yaml",
    # r"Custom_Model_cfg_3/yolo11_shuffle_Ghost_shufflev2.yaml",
    # r"Custom_Model_cfg_3/yolo11_Dwconv_Ghost_shufflev2.yaml",
    # r"Custom_Model_cfg_3/yolo11_Star_Ghost_shufflev2.yaml",
    # r"Custom_Model_cfg_3/yolo11_C3fPSC_Ghost_shufflev2.yaml",
    # r"Custom_Model_cfg_3/yolo11_C2f_LiteRepMixer_Ghost_shufflev2.yaml",
    # r"Custom_Model_cfg_3/yolo11_C3k2_LiteRep_Ghost_shufflev2.yaml",
    # "Custom_Model_cfg_3/yolo11_Dwconv_Ghost_shufflev2_CA.yaml",
    # "Custom_Model_cfg_3/yolo11m.yaml",
    "Custom_Model_cfg_3/yolo11_C3k2_Ghost_shufflev2_CA.yaml",
]

data = "Custom_dataset_cfg/vehicle_orientation.yaml"

if __name__ == '__main__':
    for model_yaml_path in model_yaml_paths:
        # 模型名（不含路径和后缀）
        model_name = os.path.splitext(os.path.basename(model_yaml_path))[0]

        print(f"\n==== Training model: {model_name} ====\n")

        model = YOLO(model_yaml_path)

        model.train(
            data=data,
            epochs=200,
            batch=128,
            imgsz=640,  # 保持不变
            cos_lr=True,
            lr0=0.02,  # ↑ 初始学习率
            lrf=0.1,  # ↑ 最终学习率比例
            close_mosaic=30,  # 提前关闭 mosaic
            save=True,
            device=[0, 1, 2, 3],
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H_%M')}"
        )

