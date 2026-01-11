# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO



# 模型配置文件
model_yaml_paths = [
                    # r"Custom_Model_cfg_3/yolo11_Ghost_Ghost_shufflev2.yaml",
                    # r"Custom_Model_cfg_3/yolo11_Pconv_Ghost_shufflev2.yaml",
                    # r"Custom_Model_cfg_3/yolo11_shuffle_Ghost_shufflev2.yaml",
                    # r"Custom_Model_cfg_3/yolo11_Dwconv_Ghost_shufflev2.yaml",
                    # r"Custom_Model_cfg_3/yolo11_Star_Ghost_shufflev2.yaml",
                    # r"Custom_Model_cfg_3/yolo11_Star_Ghost_shufflev2.yaml",
                    # r"Custom_Model_cfg_3/yolo11_C3fPSC_Ghost_shufflev2.yaml",
                    # r"Custom_Model_cfg_3/yolo11_C2f_LiteRepMixer_Ghost_shufflev2.yaml",
                    # r"Custom_Model_cfg_3/yolo11_C3k2_LiteRep_Ghost_shufflev2.yaml",
                    # "Custom_Model_cfg_3/yolo11_Dwconv_Ghost_shufflev2_CA.yaml",
                    # "Custom_Model_cfg_3/yolo11_C3k2_Ghost_shufflev2_CA.yaml",
                    # # "Custom_Model_cfg_3/yolo11m.yaml",
                    # "Custom_Model_cfg_3/yolo11_shufflev2_CA.yaml",
                    # "Custom_Model_cfg_3/yolo11_shufflev2_CA_2.yaml",
                    # "Custom_Model_cfg_3/yolo11_C3k2_Ghost_shufflev2_CA.yaml",
                    # # "Custom_Model_cfg_3/yolo11m.yaml",
                    # "Custom_Model_cfg_3/yolo11_shufflev2_CA.yaml",
                    # "Custom_Model_cfg_3/yolo11_VoVGSCSPC_Ghost_shufflev2_CA.yaml"
                    r"Custom_Model_cfg_3/yolo11_DwconvC3k2.yaml",
                    r"Custom_Model_cfg_3/yolo11_C3k2sema.yaml",
                    r'Custom_Model_cfg_3/yolo11_Dwconv_Ghost_shufflev2_CA.yaml',
                    ]

data="coco8.yaml"
#data = "Custom_dataset_cfg/coco-vehicle.yaml"
# 预训练模型
if __name__ == '__main__':

    for path in model_yaml_paths:
        model = YOLO(path)
        # results = model.train(data=data,
        #                       epochs=3,
        #                       batch=8,
        #                       imgsz=640,
        #                       cos_lr=True,
        #                       close_mosaic=50,
        #                       save= True,
        #                       device=-1,
        #                       workers=16,
        #                       name="test"+datetime.now().strftime("%Y%m%d_%H_%M"))

        print("训练完成后打印参数")

        model.info()
        # 部署准备
        model.fuse()
        model.eval()
        # model.val()
        print("训练结束打印参数")
        model.info()
