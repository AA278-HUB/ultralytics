# coding:utf-8      `
from datetime import datetime
from ultralytics import YOLO

# 模型配置文件
model_yaml_path = "Custom_Model_cfg/yolo11_dyhead.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_1.yaml"
model_yaml_path = "Custom_Model_cfg/yolo11_Ghost_Rep.yaml"
data="coco8.yaml"
data = "Custom_dataset_cfg/coco-vehicle.yaml"
# 预训练模型
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # 数据
    # modules = [
    #     'Conv', 'Conv', 'C3k2', 'Conv', 'C3k2', 'Conv', 'C3k2',
    #     'Conv', 'C3k2', 'SPPF', 'C2PSA', 'Upsample', 'Concat',
    #     'C3k2', 'Upsample', 'Concat', 'C3k2', 'Conv', 'Concat',
    #     'C3k2', 'Conv', 'Concat', 'C3k2', 'Detect'
    # ]
    #
    # params = [
    #     464, 4672, 6640, 36992, 26080, 147712, 87040,
    #     295424, 346112, 164608, 249728, 0, 0,
    #     111296, 0, 0, 32096, 36992, 0,
    #     86720, 147712, 0, 378880, 430867
    # ]
    #
    # # 过滤掉参数为0的模块（如Upsample, Concat）
    # filtered_modules = []
    # filtered_params = []
    #
    # for module, param in zip(modules, params):
    #     if param > 0:
    #         filtered_modules.append(module)
    #         filtered_params.append(param)
    #
    # # 创建图形
    # plt.figure(figsize=(14, 8))
    #
    # # 创建柱状图
    # bars = plt.bar(range(len(filtered_modules)), filtered_params, color='skyblue', edgecolor='navy')
    #
    # # 添加数值标签
    # for i, (bar, param) in enumerate(zip(bars, filtered_params)):
    #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
    #              f'{param:,}', ha='center', va='bottom', fontsize=8, rotation=45)
    #
    # # 设置图表属性
    # plt.xlabel('module', fontsize=12)
    # plt.ylabel('parameters', fontsize=12)
    # plt.title('YOLO11n module-parmaeter', fontsize=14, fontweight='bold')
    # plt.xticks(range(len(filtered_modules)), filtered_modules, rotation=45, ha='right')
    # plt.grid(axis='y', alpha=0.3)
    #
    # # 设置y轴为科学计数法显示
    # plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    #
    # # 调整布局
    # plt.tight_layout()
    #
    # # 显示图表
    # plt.show()
    #
    # # 打印统计信息
    # print(f"总模块数: {len(filtered_modules)}")
    # print(f"总参数数: {sum(filtered_params):,}")
    # print(f"\n各模块参数统计:")
    # for module in set(filtered_modules):
    #     module_params = [p for m, p in zip(filtered_modules, filtered_params) if m == module]
    #     print(f"{module}: {len(module_params)}个实例, 总参数: {sum(module_params):,}")


    model = YOLO(model_yaml_path)
    #
    results = model.train(data=data,
                          epochs=10,
                          batch=8,
                          imgsz=640,
                          cos_lr=True,
                          close_mosaic=50,
                          save= True,
                          device="0",
                          name="test"+datetime.now().strftime("%Y%m%d_%H_%M"))

