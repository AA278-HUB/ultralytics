from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import torch
import time
import gc

# 模型路径列表
models_list = [
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11n20251022_19_29\weights\best.pt",
    # r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_ghost20251023_22_20\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Rep_shufflev1_new20251028_21_48\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Rep_shufflev2_new20251031_21_38\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Ghost_Rep_shufflev1_CBAM20251031_20_37\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Ghost_Rep_shufflev2_CBAM20251101_19_03\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Ghost_Rep_shufflev1_CA_1_20251112_10_25\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Ghost_Rep_shufflev2_CA_20251116_10_17\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\MobileNetV4_20251201_11_31\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo12n_20251215_15_37\weights\best.pt"
]
data_yaml=r"Custom_dataset_cfg/test.yaml"
print("开始多模型 Benchmark 测试...")

for i, model_path in enumerate(models_list):
    print(f"\n{'=' * 50}")
    print(f"正在测试第 {i + 1} 个模型: {model_path.split('/')[-3]}")  # 打印文件夹名识别模型
    print(f"{'=' * 50}")
    # 1. 强制等待，让 GPU 降温（冷却 5 秒）
    time.sleep(5)

    # 2. 清理上一个模型的残留
    gc.collect()
    try:
        # 运行 Benchmark
        # 注意：device=0 使用 GPU，device='cpu' 使用 CPU
        # 如果 GPU 还是崩，建议先改用 device='cpu' 确认代码逻辑和自定义层是否正常
        results = benchmark(
            model=model_path,
            data=data_yaml,
            imgsz=640,
            half=False,  # 如果是 T4 GPU，可以尝试改为 True 跑 FP16 加速
            device="CPU",  # 先尝试 GPU
            format="ncnn"
        )
        print(f"✅ 模型 {i + 1} 测试完成")

    except Exception as e:
        print(f"❌ 模型 {i + 1} 测试失败，错误原因: {e}")
        # 如果 GPU 崩溃（Segment fault），try-except 可能捕获不到，系统会直接重启 Kernel
        continue

print("\n所有模型测试流程结束。")
