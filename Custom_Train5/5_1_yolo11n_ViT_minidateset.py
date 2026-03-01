# coding: utf-8
from datetime import datetime
from ultralytics import YOLO
import os

# 模型配置文件
model_yaml_paths = [
    # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1.yaml"
    "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1_ASPPF.yaml"
]
data = "Custom_dataset_cfg/vehicle_orientation_mini.yaml"

if __name__ == '__main__':
    for model_yaml_path in model_yaml_paths:
        # 模型名（不含路径和后缀）
        model_name = os.path.splitext(os.path.basename(model_yaml_path))[0]
        print(f"\n==== Training model: {model_name} ====\n")
        model = YOLO(model_yaml_path)
        # model.train(resume=True)
        Loss="CIOU"
# =======
        model.train(
            data=data,
            epochs=200,
            batch=64,
            imgsz=640,  # 保持不变
            cos_lr=True,
            lr0=0.005,  # ↑ 初始学习率
            lrf=0.2,  # ↑ 最终学习率比例
            close_mosaic=20,  # 提前关闭 mosaic
            save=True,
            device=[0,1,2,3],
            # amp=False,
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H_%M')}_loss_{Loss}"
        )
# >>>>>>> fe705cf0be66db3253a39dab4347b0f99ce842c5

