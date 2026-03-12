from multiprocessing import freeze_support

from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import torch
import time
import gc

# 模型路径列表
models_list = [
    r"C:\Users\Hunger\Desktop\ultralytics\Custom_benchmark\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\MobileNetV4_20251201_11_31\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo12n_20251215_15_37\weights\best.pt"
]
data_yaml=r"Custom_dataset_cfg/test.yaml"
# print("开始多模型 Benchmark 测试...")
if __name__ == '__main__':
    for i, model_path in enumerate(models_list):
        try:
            results = benchmark(
                model=model_path,
                data=data_yaml,
                format="torchscript",
                imgsz=640,
                half=False,  # 如果是 T4 GPU，可以尝试改为 True 跑 FP16 加速
                device=-1,  # 先尝试 GPU
            )
            print(f"✅ 模型 {i + 1} 测试完成")

        except Exception as e:
            print(f"❌ 模型 {i + 1} 测试失败，错误原因: {e}")
            continue

