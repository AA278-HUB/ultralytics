from datetime import datetime
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

if __name__ == '__main__':
    # 教师模型（只加载一次，后面复用）
    # teacher = YOLO(r"C:\Users\Hunger\Desktop\ultralytics\Custom_Distiller\best_m.pt")
    teacher = YOLO(r"/sysv/vehicle_orientation_mini/ultralytics/Custom_Distiller/best_m.pt")
    student_path="yolo11n.yaml"
    # teacher.model.eval()           # 可以在这里冻结，也可以放在 trainer 里处理
    # for p in teacher.model.parameters():
    #     p.requires_grad = False

    # 要测试的 DistillWeight 值（建议从较宽范围开始，后面再细化）
    distill_weights_to_test = [1.0]

    # 数据集和基本配置（共用部分）
    common_args = dict(
        data="Custom_dataset_cfg/vehicle_orientation_mini.yaml",
        epochs=200,
        batch=32,
        imgsz=640,
        cos_lr=True,
        lr0=0.01,
        lrf=0.1,
        close_mosaic=20,
        save=True,
        device=0,
        # amp=True,                # 建议开启，除非显存/稳定性有问题
        Distill=True,
        Teacher=teacher.model,
        distill_loss="cwd",      # 或 "mse", "l1", "cwd" 等，看你实现的是哪个
    )

    print(f"将测试 {len(distill_weights_to_test)} 个 DistillWeight 值")

    for idx, dw in enumerate(distill_weights_to_test, 1):
        # print(f"\n{'='*60}")
        # print(f"开始第 {idx}/{len(distill_weights_to_test)} 次实验")
        # print(f"DistillWeight = {dw:.1f}")
        # print(f"{'='*60}\n")

        name = f"yolo11n_distill_dw{dw:.1f}_{datetime.now().strftime('%Y%m%d_%H%M')}"

        try:
            student = YOLO(student_path)  # 每次新建 student，避免权重污染

            student.train(
                **common_args,
                name=name,
                DistillWeight=dw,      # 这里传入你自定义的参数
            )
        except Exception as e:
            print(f"训练过程中发生错误：{e}")
            continue

    print("\n所有 DistillWeight 实验已完成")