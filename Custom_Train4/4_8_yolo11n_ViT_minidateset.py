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
    # #             r"Custom_Model_cfg_7/yolo11_MAFPN_modifyX_C3k2.yaml"
    # r"Custom_Model_cfg_7/yolo11_Ghost_Rep_Ghost_shufflev2_MAFPN.yaml",
    # r"Custom_Model_cfg_7/yolo11_Ghost_Rep_Ghost_shufflev2_CA_MAFPN.yaml",
    # r"Custom_Model_cfg_7/yolo11_MAFPN_modifyX_Repvit.yaml",
    # r"Custom_Model_cfg_7/yolo11-C3k2-iRMB.yaml",
    # r"Custom_Model_cfg_7/yolo11-C3k2-MambaOut.yaml",
    # r"Custom_Model_cfg_7/yolo11-C3k2-Faster-EMA.yaml",
    # r"Custom_Model_cfg_7/yolo11-C3k2-Star-CAA.yaml",
    # "Custom_Model_cfg_7/yolo11n.yaml"
    # r"/sysv/vehicle_orientation_mini/ultralytics/runs/detect/yolo11x_20260128_13_26/weights/last.pt"
    #  "Custom_Model_cfg_7/yolo11_RepStar.yaml"
    # "Custom_Model_cfg_7/yolo11-C3k2-LSBlock.yaml",

# 可用模型，按照可能的效果排序
# "Custom_Model_cfg_7/yolo11-C2CGA.yaml",                 # 可能是比较基础的架构
# "Custom_Model_cfg_7/yolo11-C2PSA-CGLU.yaml",  # 更复杂的网络可能具有较高的性能
# "Custom_Model_cfg_7/yolo11-C2PSA-DYT.yaml",  # 相对较新的结构，应该有效
# "Custom_Model_cfg_7/yolo11-C2PSA-EDFFN.yaml",  # EDFFN 可能会有不错的表现
# "Custom_Model_cfg_7/yolo11-C2PSA-FMFFN.yaml",  # FMFFN 模型也比较常见，性能较强



# "Custom_Model_cfg_7/yolo11-C2PSA-Mona.yaml",  # 这个模型可能较有潜力
# "Custom_Model_cfg_7/yolo11-C2PSA-SEFFN.yaml",  # SEFFN 是一种新的结构，可能有一定提升
"Custom_Model_cfg_7/yolo11-C2PSA-SEFN.yaml",  # 也许是针对某些情况优化的模型

# "Custom_Model_cfg_7/yolo11-C2Pola-DYT.yaml",  # 暂时不行
# "Custom_Model_cfg_7/yolo11-C2Pola.yaml",       # 暂时不行


"Custom_Model_cfg_7/yolo11-C2TSSA-DYT-Mona-EDFFN.yaml",  # 相对复杂的 TSSA 模型，加入 EDFFN 后的性能可能较强
"Custom_Model_cfg_7/yolo11-C2TSSA-DYT-Mona-SEFFN.yaml",  # SEFFN 加入后，也许能有更好的效果
"Custom_Model_cfg_7/yolo11-C2TSSA-DYT-Mona-SEFN.yaml",   # SEFN 可能对某些任务有改善
"Custom_Model_cfg_7/yolo11-C2TSSA-DYT-Mona.yaml",        # DYT-Mona 可能会表现较好
"Custom_Model_cfg_7/yolo11-C2TSSA-DYT.yaml",             # 相对复杂的模型，但可能需要调优
"Custom_Model_cfg_7/yolo11-C2TSSA.yaml",                 # TSSA 的基础版本，效果不错

# "Custom_Model_cfg_7/yolo11-C2ASSA.yaml",               # 暂时不行

"Custom_Model_cfg_7/yolo11-C2BRA.yaml",                 # 比较基础的结构，可能表现一般
"Custom_Model_cfg_7/yolo11-C2DA.yaml",                  # 基本架构，但缺乏高效优化
# "Custom_Model_cfg_7/yolo11-C2DPB.yaml",                 # 暂时不行


            ]
data = "Custom_dataset_cfg/vehicle_orientation_mini.yaml"

if __name__ == '__main__':
    for model_yaml_path in model_yaml_paths:
        # 模型名（不含路径和后缀）
        model_name = os.path.splitext(os.path.basename(model_yaml_path))[0]
        print(f"\n==== Training model: {model_name} ====\n")
        model = YOLO(model_yaml_path)
        # model.train(resume=True)
#
# <<<<<<< HEAD
#         model.train(
#             data=data,
#             epochs=200,
#             batch=32,
#             imgsz=640,  # 保持不变
#             cos_lr=True,
#             lr0=0.01,  # ↑ 初始学习率
#             lrf=0.1,  # ↑ 最终学习率比例
#             close_mosaic=20,  # 提前关闭 mosaic
#             save=True,
#             device=[0,1,2,3],
#             amp=False,
#             name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H_%M')}"
#         )
# =======
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
            device=[0,1,2,3],
            # amp=False,
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H_%M')}"
        )
# >>>>>>> fe705cf0be66db3253a39dab4347b0f99ce842c5

