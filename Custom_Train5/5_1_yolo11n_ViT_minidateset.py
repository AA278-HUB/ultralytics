# coding: utf-8
from datetime import datetime
from ultralytics import YOLO
import os

# 模型配置文件
model_yaml_paths = [
    # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1.yaml"
    # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1_GSPPF.yaml"
    # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1_ESPPF.yaml",
    # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1_SSPPF.yaml",
    # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1_DSPPF.yaml",
    # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1_Dy_SPPF.yaml",
    # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1_L_SPPF.yaml",
    # "Custom_Model_cfg_14/yolo11_MAFPN_modifyX_uniRepLK_v2.yaml",
    # "Custom_Model_cfg_14/yolo11_MAFPN_modifyX_uniRepLKv5_v1.yaml",
    # "Custom_Model_cfg_15/yolo11_MAFPN_modifyX_uniRepLKv5_v2.yaml",
    # "Custom_Model_cfg_15/yolo11_MAFPN_modifyX_uniRepLKv5_v2_GSPPF.yaml",
    "Custom_Model_cfg_15/yolo11_MAFPN_modifyX_uniRepLKv5_v3.yaml",

]
cls_type="None"
full_iou_name="CIoU"
nwd_on=False
use_wise_framework=False


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
            cls_loss_type=cls_type,  # 传给分类损失
            iou_type=full_iou_name,  # 传给 BboxLoss
            nwd_loss=nwd_on,  # 是否开启 NWD
            use_wise_framework=use_wise_framework,
            iou_ratio=0.5 if nwd_on else 1.0,  # NWD 占比
            # amp=False,
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H_%M')}_loss_{Loss}"
        )
# >>>>>>> fe705cf0be66db3253a39dab4347b0f99ce842c5

