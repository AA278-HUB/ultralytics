# coding: utf-8
from datetime import datetime
from ultralytics import YOLO
import os

# 模型配置文件
model_yaml_paths = [
           # r"Custom_Model_cfg_4/yolo11n.yaml",
           # r"Custom_Model_cfg_4/yolo11_bifpn.yaml",
           # r"Custom_Model_cfg_4/yolo11_bifpn_2.yaml",
           # r"Custom_Model_cfg_4/yolo11_C3RepGhost.yaml",
           # r"Custom_Model_cfg_4/yolo11_DCNv4.yaml"
           #  "Custom_Model_cfg_4/yolo11_FFCM.yaml",
           #  r"Custom_Model_cfg_4/yolo11_FFCM_2.yaml"
           #    r"Custom_Model_cfg_4/yolo11_bifpn_3.yaml",
           # #      r"Custom_Model_cfg_4/yolo11_ASFF_2.yaml"
           #  r"Custom_Model_cfg_5/yolo11_FD.yaml",
           #  r"Custom_Model_cfg_5/yolo11_ASFF_2_Dysample.yaml",
           #  r"Custom_Model_cfg_5/yolo11_RepViTblock.yaml"
           #  r"Custom_Model_cfg_5/yolo11_Mamba.yaml"
           #  r"Custom_Model_cfg_5/yolo11-C3k2-MogaBlock.yaml"
           #    r"Custom_Model_cfg_5/yolo11-C3k2-MambaVision.yaml",
              # r"Custom_Model_cfg_5/yolo11-MAFPN.yaml",
            #  "Custom_Model_cfg_6/yolo11_MAFPN_RepVit_2.yaml",
            # "Custom_Model_cfg_6/yolo11_MAFPN_Dysample.yaml"
    # r"Custom_Model_cfg_6/yolo11_RepHMS_ASFF2.yaml",
    #               "Custom_Model_cfg_7/yolo11_MAFPN_dyhead.yaml",
    #             r"Custom_Model_cfg_7/yolo11_MAFPN_modify.yaml",
    #             "Custom_Model_cfg_7/yolo11_MAFPN_modify_C3k2.yaml",
                r"Custom_Model_cfg_7/yolo11_MAFPN_modifyX_C3k2.yaml"

            ]
data = "Custom_dataset_cfg/vehicle_orientation_mini.yaml"

if __name__ == '__main__':
    for model_yaml_path in model_yaml_paths:
        # 模型名（不含路径和后缀）
        model_name = os.path.splitext(os.path.basename(model_yaml_path))[0]
        print(f"\n==== Training model: {model_name} ====\n")
        model = YOLO(model_yaml_path)

        model.train(
            data=data,
            epochs=200,
            batch=64,
            imgsz=640,  # 保持不变
            cos_lr=True,
            lr0=0.01,  # ↑ 初始学习率
            lrf=0.1,  # ↑ 最终学习率比例
            close_mosaic=20,  # 提前关闭 mosaic
            save=True,
            device=-1,
            amp=False,
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H_%M')}"
        )

