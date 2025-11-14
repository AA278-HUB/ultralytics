from ultralytics import YOLO

# 模型路径列表
model_path = [
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11n20251022_19_29\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_ghost20251023_22_20\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Rep_shufflev1_new20251028_21_48\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Ghost_Rep_shufflev1_CBAM20251031_20_37\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Rep_shufflev2_new20251031_21_38\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Ghost_Rep_shufflev2_CBAM20251101_19_03\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Ghost_Rep_shufflev1_CA_20251110_14_44\weights\best.pt",
    r"C:\Users\Hunger\Desktop\实验数据_new\yolo11_Ghost_Rep_shufflev1_CA_1_20251112_10_25\weights\best.pt"
]

data = r"Custom_dataset_cfg/vehicle_orientation.yaml"

if __name__ == '__main__':
    for path in model_path:
        print(f"\n正在验证模型: {path}")
        model = YOLO(path)
        # 模型推理阶段自动 fuse，因此手动 fuse 可省略
        # model.fuse()
        model.eval()
        metrics = model.val(data=data,batch=32,device="0")  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category

# import os
# import shutil
# import random
# from pathlib import Path
#
#
# def create_sampled_dataset_split():
#     # 原始数据集配置
#     original_path = r"E:\datasets\vehicle_orientation"
#     original_images_train = os.path.join(original_path, "images/train")
#     original_labels_train = os.path.join(original_path, "labels/train")
#
#     # 新数据集配置
#     new_path = r"E:\datasets\vehicle_orientation_test"
#     new_images_train = os.path.join(new_path, "images/train")
#     new_labels_train = os.path.join(new_path, "labels/train")
#     new_images_val = os.path.join(new_path, "images/val")
#     new_labels_val = os.path.join(new_path, "labels/val")
#
#     # 创建新数据集目录结构
#     os.makedirs(new_images_train, exist_ok=True)
#     os.makedirs(new_labels_train, exist_ok=True)
#     os.makedirs(new_images_val, exist_ok=True)
#     os.makedirs(new_labels_val, exist_ok=True)
#
#     # 获取所有标签文件
#     label_files = [f for f in os.listdir(original_labels_train) if f.endswith('.txt')]
#     print(f"原始训练集标签文件数量: {len(label_files)}")
#
#     # 随机抽取1000个标签文件
#     if len(label_files) < 1000:
#         print(f"警告: 原始训练集只有 {len(label_files)} 个标签文件，少于1000个")
#         sampled_label_files = label_files
#     else:
#         sampled_label_files = random.sample(label_files, 1000)
#
#     # 按4:6比例分割为训练集和验证集
#     split_index = int(len(sampled_label_files) * 0.4)  # 40% 训练集
#     train_label_files = sampled_label_files[:split_index]
#     val_label_files = sampled_label_files[split_index:]
#
#     print(f"新数据集分割: 训练集 {len(train_label_files)} 个样本, 验证集 {len(val_label_files)} 个样本")
#
#     # 复制训练集样本
#     train_count = 0
#     for label_file in train_label_files:
#         # 复制标签文件
#         src_label_path = os.path.join(original_labels_train, label_file)
#         dst_label_path = os.path.join(new_labels_train, label_file)
#         shutil.copy2(src_label_path, dst_label_path)
#
#         # 查找并复制对应的图片文件
#         label_name = os.path.splitext(label_file)[0]
#         image_found = False
#
#         # 尝试不同的图片扩展名
#         for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
#             src_image_path = os.path.join(original_images_train, label_name + ext)
#             if os.path.exists(src_image_path):
#                 dst_image_path = os.path.join(new_images_train, label_name + ext)
#                 shutil.copy2(src_image_path, dst_image_path)
#                 train_count += 1
#                 image_found = True
#                 break
#
#         if not image_found:
#             print(f"警告: 找不到标签文件 {label_file} 对应的图片")
#
#     # 复制验证集样本
#     val_count = 0
#     for label_file in val_label_files:
#         # 复制标签文件
#         src_label_path = os.path.join(original_labels_train, label_file)
#         dst_label_path = os.path.join(new_labels_val, label_file)
#         shutil.copy2(src_label_path, dst_label_path)
#
#         # 查找并复制对应的图片文件
#         label_name = os.path.splitext(label_file)[0]
#         image_found = False
#
#         # 尝试不同的图片扩展名
#         for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
#             src_image_path = os.path.join(original_images_train, label_name + ext)
#             if os.path.exists(src_image_path):
#                 dst_image_path = os.path.join(new_images_val, label_name + ext)
#                 shutil.copy2(src_image_path, dst_image_path)
#                 val_count += 1
#                 image_found = True
#                 break
#
#         if not image_found:
#             print(f"警告: 找不到标签文件 {label_file} 对应的图片")
#
#     print(f"成功创建新数据集:")
#     print(f"  训练集: {train_count} 张图片, {len(train_label_files)} 个标签文件")
#     print(f"  验证集: {val_count} 张图片, {len(val_label_files)} 个标签文件")
#
#     # 创建数据集配置文件
#     config_content = f"""# 数据集根目录路径
# path: {new_path}
# # 训练集路径
# train: images/train
# ## 验证集路径
# val: images/val
#
# # 类别数量
# nc: 4
# # 类别名称列表
# names: ['car', 'motorcycle', 'bus', 'truck']"""
#
#     config_path = os.path.join(new_path, "data.yaml")
#     with open(config_path, 'w', encoding='utf-8') as f:
#         f.write(config_content)
#
#     print(f"数据集配置文件已创建: {config_path}")
#
#     # 显示各类别在训练集和验证集中的分布
#     def count_class_distribution(label_files, labels_dir):
#         distribution = {}
#         for label_file in label_files:
#             label_path = os.path.join(labels_dir, label_file)
#             try:
#                 with open(label_path, 'r') as f:
#                     for line in f:
#                         class_id = int(line.strip().split()[0])
#                         class_name = ['car', 'motorcycle', 'bus', 'truck'][class_id]
#                         distribution[class_name] = distribution.get(class_name, 0) + 1
#             except Exception as e:
#                 print(f"读取标签文件 {label_file} 时出错: {e}")
#         return distribution
#
#     print("\n训练集类别分布:")
#     train_dist = count_class_distribution(train_label_files, new_labels_train)
#     for class_name, count in train_dist.items():
#         print(f"  {class_name}: {count} 个标注")
#
#     print("\n验证集类别分布:")
#     val_dist = count_class_distribution(val_label_files, new_labels_val)
#     for class_name, count in val_dist.items():
#         print(f"  {class_name}: {count} 个标注")
#
#
# if __name__ == "__main__":
#     create_sampled_dataset_split()