# coding: utf-8
from datetime import datetime
from ultralytics import YOLO
import os
import itertools

# --- 1. 配置损失函数实验网格 ---
# 分类损失映射 (Key: 传给Loss代码的参数, Value: 用于文件命名的缩写)
CLS_MAP = {
    # "QualityfocalLoss": "Qualityfocal",
    # "EMASlideLoss": "EMASlide",
    # "FocalLoss": "Focal",
}
#"VarifocalLoss": "Varifocal", "SlideLoss": "Slide",

# 定位损失 (Base IoU)
IOU_TYPES = ["CIoU","D-InterpIoU","InterpIoU","GIoU", "DIoU", "EIoU", "SIoU", "ShapeIoU", "WIoU", "PIoU", "PIoU2", "Inner_MPDIoU", "MPDIoU",
             "Focaler_MPDIoU","alpha_IoU"]  #


# 增强插件 (None: 保持原样, Inner: 内部辅助框, Focaler: 难样本聚焦)
ENHANCE_TYPES = ["None",]  # "Inner", "Focaler"

# NWD 小目标插件配置 (True/False)
NWD_OPTIONS = [False]#True,
# Wise框架
use_wise_framework = [False] #True,



# =========实验==========

IOU_TYPES = ["Inner_MPDIoU", "MPDIoU",
             "Focaler_MPDIoU","CIoU","PIoU", "PIoU2",] #"CIoU", "PIoU", "PIoU2", #  "PIoU", "PIoU2", "WIoU", "Inner_MPDIoU", "MPDIoU", "Focaler_MPDIoU"
IOU_TYPES=["CIoU"]
CLS_MAP = {
    # "QualityfocalLoss": "Qualityfocal",
    # "EMASlideLoss": "EMASlide",
    "Fuck": "Fuck",
}

use_wise_framework = [False]
# 增强插件 (None: 保持原样, Inner: 内部辅助框, Focaler: 难样本聚焦)
ENHANCE_TYPES = ["None"]  #, "Focaler"
NWD_OPTIONS = [False]
# 模型配置文件
model_yaml_paths = [
                            # "Custom_Model_cfg_14/yolo11_MAFPN_modifyX_uniRepLK_v2.yaml",
                            # "Custom_Model_cfg_13/yolo11_MAFPN_modifyX_uniRepLK_v1.yaml",
                            # "Custom_Model_cfg_15/yolo11_MAFPN_modifyX_uniRepLKv5_v2.yaml
                            # "Custom_Model_cfg_15/yolo11_MAFPN_modifyX_uniRepLKv5_v3.yaml",
    # "Custom_Model_cfg/yolov5.yaml",
    # "Custom_Model_cfg/yolov6.yaml",
    # "Custom_Model_cfg/yolov8.yaml",
    # "Custom_Model_cfg/yolov9t.yaml",
    # "Custom_Model_cfg/yolov10n.yaml",
    "Custom_Model_cfg/yolo12n.yaml",
    "Custom_Model_cfg/MobileNetV4.yaml",

]


data = "Custom_dataset_cfg/vehicle_orientation_mini.yaml"
# data = "coco8.yaml"

MPDIoU_Count = 0
if __name__ == '__main__':
    # 生成所有实验组合的笛卡尔积
    # 组合顺序：(分类, IoU基准, 增强方式, NWD开关)
    experiments = list(itertools.product(CLS_MAP.keys(), IOU_TYPES, ENHANCE_TYPES, use_wise_framework, NWD_OPTIONS))

    for model_yaml_path in model_yaml_paths:
        model_name = os.path.splitext(os.path.basename(model_yaml_path))[0]

        for cls_type, iou_base, enhance, use_wise_framework, nwd_on in experiments:
            # --- 2. 构造符合规范的文件名字符串 ---
            # 处理 Inner- 或 Focaler- 前缀`
            if "MPDIoU" in iou_base:
                if use_wise_framework is not False or enhance != "None":
                    continue

                # use_wise_framework = False
                # enhance = "None"
                # if MPDIoU_Count >= 3:
                #     continue
                # MPDIoU_Count += 1
            if "WIoU" in iou_base:
                enhance = "None"
                if not use_wise_framework:
                    continue
            # if cls_type == "EMASlide" and iou_base == "CIoU":
            #     continue
            # if enhance == "None" and use_wise_framework == False:
            #     if iou_base == "CIoU" or iou_base == "SIoU":
            #         continue

            full_iou_name = iou_base if enhance == "None" else f"{enhance}_{iou_base}"
            nwd_suffix = "NWD_On" if nwd_on else "NWD_Off"
            wise_suffix = "WISE_On" if use_wise_framework else "WISE_Off"
            # 最终实验标识符 (例如: Focal__Inner_WIoUv3__NWD_On)
            loss_exp_name = f"{CLS_MAP[cls_type]}__{full_iou_name}__{wise_suffix}__{nwd_suffix}"

            print(f"\n" + "=" * 50)
            print(f"🚀 正在训练组合: {loss_exp_name}")
            print(f"=" * 50 + "\n")

            # 初始化模型
            model = YOLO(model_yaml_path)

            # --- 3. 启动训练 ---
            # 注意：需要在 ultralytics/utils/loss.py 中读取这些自定义参数
            model.train(
                data=data,
                epochs=200,
                batch=64,
                imgsz=640,
                cos_lr=True,
                lr0=0.005,
                lrf=0.2,
                close_mosaic=20,
                save=True,
                device=[0, 1, 2, 3],
                # 动态命名：模型名 + 时间 + 损失组合
                name=f"{model_name}_{datetime.now().strftime('%m%d%H%M')}_{loss_exp_name}",
                # 以下是自定义传参（确保你的 loss.py 有解析逻辑）
                # 如果是官方代码，通常需要通过 cfg 文件或修改 v8DetectionLoss 接收
                cls_loss_type=cls_type,  # 传给分类损失
                iou_type=full_iou_name,  # 传给 BboxLoss
                nwd_loss=nwd_on,  # 是否开启 NWD
                use_wise_framework=use_wise_framework,
                iou_ratio=0.5 if nwd_on else 1.0  # NWD 占比
            )
            # 写入日志文件（追加模式）
            # with open("LOG_FILE_Loss.txt", "a", encoding="utf-8") as f:
            #     f.write(f"{loss_exp_name}\n")
