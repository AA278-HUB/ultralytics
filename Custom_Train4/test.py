# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO


model_yaml_paths = [
    # "Custom_Model_cfg_4/yolo11n.yaml",
    # "Custom_Model_cfg_4/yolo11_bifpn.yaml",
    # "Custom_Model_cfg_4/yolo11_bifpn_2.yaml",
    # "Custom_Model_cfg_4/yolo11_C3RepGhost.yaml",
    # "Custom_Model_cfg_4/yolo11_Star_Ghost_shufflev2.yaml",
    # "Custom_Model_cfg_4/yolo11_Dwconv_Ghost_shufflev2.yaml",
    # "Custom_Model_cfg_4/yolo11_RepViTBlock_Ghost_shufflev2.yaml",
    # "Custom_Model_cfg_4/yolo11_C3k2sema.yaml",
    # "Custom_Model_cfg_4/yolo11_Pconv_Ghost_shufflev2.yaml",
    # "Custom_Model_cfg_4/yolov8n_gold_yolo_neck_v3.yaml",
    # "Custom_Model_cfg_4/yolo11_gold_Neck.yaml",
    # "Custom_Model_cfg_4/yolo11_DCNv4.yaml",
    # "Custom_Model_cfg_4/yolo11_FFCM.yaml",
    # "Custom_Model_cfg_4/yolo11_bifpn_3.yaml",
    # "Custom_Model_cfg_4/yolo11_ASFF.yaml",
    # "Custom_Model_cfg_4/yolo11_ASFF_2.yaml",
    # "Custom_Model_cfg_4/yolov12n-MAFPN.yaml",
    # "Custom_Model_cfg_4/yolo11_MAFPN.yaml",
    # "Custom_Model_cfg_4/yolo11_Slim_Neck.yaml",
    # "Custom_Model_cfg_5/yolo11_FD.yaml",
    # "Custom_Model_cfg_5/yolo11_ASFF_2_Dysample.yaml",
    # "Custom_Model_cfg_5/yolo11_RepViTblock.yaml",
    # "Custom_Model_cfg_5/yolo11-C3k2-MogaBlock.yaml",
    # "Custom_Model_cfg_6/yolo11_Ghost_MAFPN.yaml",
    # "Custom_Model_cfg_6/yolo11_Ghost_C3k2_MAFPN.yaml",
    # "Custom_Model_cfg_6/yolo11-C3k2-UniRepNetBlock.yaml",
    # "Custom_Model_cfg_6/yolo11_RepHMS_ASFF2.yaml",
    # "Custom_Model_cfg_5/yolo11-RepHMS.yaml",
    # "Custom_Model_cfg_6/yolo11-MAFPN_RepVit_2.yaml",
    # "Custom_Model_cfg_6/yolo11-MAFPN_Dysample.yaml",
    # "Custom_Model_cfg_7/yolo11_MAFPN_modify.yaml",
    # "Custom_Model_cfg_7/yolo11_MAFPN_modifyX_C3k2.yaml",
    # "Custom_Model_cfg_7/yolo11_Ghost_Rep_Ghost_shufflev2_MAFPN.yaml",
    # "Custom_Model_cfg_7/yolo11_Ghost_Rep_Ghost_shufflev2_CA_MAFPN.yaml",
    # "Custom_Model_cfg_7/yolo11-C3k2-iRMB.yaml",
    # "Custom_Model_cfg_7/yolo11-C3k2-MambaOut.yaml",
    # "Custom_Model_cfg_7/yolo11-C3k2-Faster-EMA.yaml",
    # "Custom_Model_cfg_7/yolo11-C3k2-Star-CAA.yaml",
    # "Custom_Model_cfg_7/yolo11_RepStar.yaml",
    # "Custom_Model_cfg_7/yolo11-C3k2-LSBlock.yaml",

    # 新文件添加了前缀 "Custom_Model_cfg_7/"
    # "Custom_Model_cfg_7/yolo11-C2ASSA.yaml", #暂时不行
    # "Custom_Model_cfg_7/yolo11-C2BRA.yaml",
    # "Custom_Model_cfg_7/yolo11-C2CGA.yaml",
    # "Custom_Model_cfg_7/yolo11-C2DA.yaml",
    # "Custom_Model_cfg_7/yolo11-C2DPB.yaml",      #暂时不行
    # "Custom_Model_cfg_7/yolo11-C2Pola-DYT.yaml", #暂时不行
    # "Custom_Model_cfg_7/yolo11-C2Pola.yaml",       #暂时不行
    # "Custom_Model_cfg_7/yolo11-C2PSA-CGLU.yaml",
    # "Custom_Model_cfg_7/yolo11-C2PSA-DYT.yaml",
    # "Custom_Model_cfg_7/yolo11-C2PSA-EDFFN.yaml",
    # "Custom_Model_cfg_7/yolo11-C2PSA-FMFFN.yaml", #暂时不行
    # "Custom_Model_cfg_7/yolo11-C2PSA-Mona.yaml",
    # "Custom_Model_cfg_7/yolo11-C2PSA-SEFFN.yaml",
    # "Custom_Model_cfg_7/yolo11-C2PSA-SEFN.yaml", #暂时不行
    # "Custom_Model_cfg_7/yolo11-C2TSSA-DYT-Mona-EDFFN.yaml",
    # "Custom_Model_cfg_7/yolo11-C2TSSA-DYT-Mona-SEFFN.yaml",
    # "Custom_Model_cfg_7/yolo11-C2TSSA-DYT-Mona-SEFN.yaml",#暂时不行
    # "Custom_Model_cfg_7/yolo11-C2TSSA-DYT-Mona.yaml",
    # "Custom_Model_cfg_7/yolo11-C2TSSA-DYT.yaml",
    # "Custom_Model_cfg_7/yolo11-C2TSSA.yaml"
    # "Custom_Model_cfg_8/yolo11-RepHMS_plus.yaml",
    # "Custom_Model_cfg_8/yolo11_MAFPN_plus.yaml",
    # "Custom_Model_cfg_4/yolo11_MAFPN.yaml",
    # "Custom_Model_cfg_9/yolo11_MAFPN_modifyX_C3k2.yaml"

    # "Custom_Model_cfg_9/yolo11-RepHMS_V2.yaml"


    # "Custom_Model_cfg_9/yolo11-RepDGM.yaml",
    # "Custom_Model_cfg_9/yolo11-RepDGM_V2.yaml",
    # "Custom_Model_cfg_9/yolo11-RepGMS.yaml",
    # "Custom_Model_cfg_9/yolo11-RepGVA_ELAN.yaml",
    # "Custom_Model_cfg_9/yolo11-RepSFA.yaml",
    # "Custom_Model_cfg_9/yolo11-RepHMS_pro.yaml",
    # "Custom_Model_cfg_8/yolo11-RepHMS_plus.yaml",
    # "Custom_Model_cfg_9/yolo11-RepHMA.yaml",

    # "Custom_Model_cfg_10/yolo11_MAFPN_modifyX_RepDGM.yaml",
    # "Custom_Model_cfg_10/yolo11_MAFPN_modifyX_RepDGMv2.yaml",
    # "Custom_Model_cfg_10/yolo11_MAFPN_modifyX_RepDGM_GMS.yaml",
    #
    # "Custom_Model_cfg_10/yolo11_MAFPN_modifyX_RepDGM_GVA_ELAN.yaml",
    # "Custom_Model_cfg_10/yolo11_MAFPN_modifyX_RepHMA.yaml",
    #
    # "Custom_Model_cfg_10/yolo11_MAFPN_modifyX_RepHMS_pro.yaml",
    # "Custom_Model_cfg_10/yolo11_MAFPN_modifyX_RepSFA.yaml",


    # "Custom_Model_cfg_11/yolo11_MAFPN_modifyX_RepHMS_Gemini.yaml"
    # "Custom_Model_cfg_11/yolo11_MAFPN_modifyX_RepHMS_Gemini_Star.yaml",
    # "Custom_Model_cfg_11/yolo11_MAFPN_modifyX_RepHMS_Grok.yaml",
    # "Custom_Model_cfg_11/yolo11_MAFPN_modifyX_RepHMS_Gemini_LightStar.yaml"
    # "Custom_Model_cfg_11/yolo11_MAFPN_modifyX_RepHMS_Grok_V2.yaml"
    # "Custom_Model_cfg_11/yolo11_MAFPN_modifyX_RepHMS_Grok_V3.yaml"
    # "Custom_Model_cfg_11/yolo11_MAFPN_modifyX_RepHMS_Omni.yaml"
    # "Custom_Model_cfg_11/yolo11_MAFPN_modifyX_RepHMS_GeminiV2.yaml"
    # "Custom_Model_cfg_12/yolo11_MAFPN_modifyX_RepHMS.yaml",
    # "Custom_Model_cfg_12/yolo11_MAFPN_modifyX_RepHMS_Star.yaml",


    # "Custom_Model_cfg_12/yolo11_MAFPN_modifyX_RepHMS_Ultra.yaml"
    # "Custom_Model_cfg_12/yolo11_MAFPN_modifyX_RepHMS_Omni.yaml "
    # "Custom_Model_cfg_12/yolo11_MAFPN_modifyX_RepHMS_UltraGrok.yaml"

    # "Custom_Model_cfg_12/yolo11_Star.yaml",
    # "Custom_Model_cfg_12/yolo11_PKI.yaml",
    # "Custom_Model_cfg_12/yolo11_SC.yaml",

    # "Custom_Model_cfg_12/yolo11_StarLK.yaml",
    # "Custom_Model_cfg_12/yolo11_C3k2_StarDynamic.yaml",
    # "Custom_Model_cfg_12/yolo11_C3k2_HIE.yaml"
    # "Custom_Model_cfg_12/yolo11_UniStar.yaml",
    # "Custom_Model_cfg_12/yolo11_SGLK.yaml"
    # "Custom_Model_cfg_12/yolo11_DBSGLK.yaml"
    # "Custom_Model_cfg_12/yolo11_HSG.yaml"
    # "Custom_Model_cfg_12/yolo11_SGLK_V5.yaml"
    # "Custom_Model_cfg_12/yolo11_MSDA.yaml"
    # "Custom_Model_cfg_12/yolo11_UniRep.yaml"
    "Custom_Model_cfg_12/yolo11_StarRepLK.yaml",
    # "Custom_Model_cfg_12/yolo11_AdvUniRepLK.yaml",

]





data="coco8.yaml"
#data = "Custom_dataset_cfg/coco-vehicle.yaml"
# 预训练模型
if __name__ == '__main__':

    for path in model_yaml_paths:
        model = YOLO(path)
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
        # model.eval()
        # model.val()
        # print("训练结束打印参数")
        # model.info()
