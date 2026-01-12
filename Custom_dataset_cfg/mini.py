# ===================== 引入库 =====================
import os
import random
import shutil
from collections import defaultdict

# ===================== 配置区 =====================

# 原始数据集根目录
SRC_ROOT = r"E:\datasets\vehicle_orientation"

# 新 mini 数据集根目录
DST_ROOT = r"E:\datasets\vehicle_orientation_mini"

# 图片后缀名（如果是 png 改这里）
IMG_EXT = ".jpg"

# mini 数据集占原始数据集比例
SAMPLE_RATIO = 0.25

# motorcycle 的类别 id（['car', 'motorcycle', 'bus', 'truck']）
MOTORCYCLE_CLS = 1

# mini 数据集 train / val 划分比例
TRAIN_RATIO = 0.8

# 固定随机种子，保证每次结果一致
SEED = 42
random.seed(SEED)

# ===================== 工具函数 =====================

def mkdir(path):
    """如果目录不存在就创建"""
    os.makedirs(path, exist_ok=True)

# ===================== 原始数据路径 =====================

src_img_train = os.path.join(SRC_ROOT, "images", "train")
src_lbl_train = os.path.join(SRC_ROOT, "labels", "train")

src_img_val = os.path.join(SRC_ROOT, "images", "val")
src_lbl_val = os.path.join(SRC_ROOT, "labels", "val")

# ===================== mini 数据路径 =====================

dst_img_train = os.path.join(DST_ROOT, "images", "train")
dst_lbl_train = os.path.join(DST_ROOT, "labels", "train")

dst_img_val = os.path.join(DST_ROOT, "images", "val")
dst_lbl_val = os.path.join(DST_ROOT, "labels", "val")

# 创建 mini 目录结构
for d in [dst_img_train, dst_lbl_train, dst_img_val, dst_lbl_val]:
    mkdir(d)

# ===================== Step 1：统计 train + val =====================

# class_id -> 包含该类别的图片集合
class_to_imgs = defaultdict(set)

# 所有图片集合（train + val）
all_images = set()

def collect_from(label_dir):
    """从某个 labels 目录中统计图片和类别"""
    for lbl_file in os.listdir(label_dir):
        if not lbl_file.endswith(".txt"):
            continue

        img_name = lbl_file.replace(".txt", IMG_EXT)
        all_images.add(img_name)

        with open(os.path.join(label_dir, lbl_file), "r") as f:
            for line in f:
                cls = int(line.split()[0])
                class_to_imgs[cls].add(img_name)

# 统计 train 和 val
collect_from(src_lbl_train)
collect_from(src_lbl_val)

# ===================== Step 2：计算目标总量 =====================

total_original = len(all_images)
target_total = int(total_original * SAMPLE_RATIO)

print("原始总图片数：", total_original)
print("mini 目标总图片数（1/4）：", target_total)

# ===================== Step 3：抽样（弱类保护） =====================

# 先把 motorcycle 全部保留
selected_imgs = set(class_to_imgs[MOTORCYCLE_CLS])
motor_count = len(selected_imgs)

print("motorcycle 全保留数量：", motor_count)

# 还能再选多少张
remaining_quota = target_total - motor_count

# 如果 motorcycle 已经超过 1/4（理论上很少见）
if remaining_quota <= 0:
    selected_imgs = set(
        random.sample(list(selected_imgs), target_total)
    )
else:
    # 其他类别的图片（去掉 motorcycle 已经选过的）
    other_imgs = set()

    for cls, imgs in class_to_imgs.items():
        if cls == MOTORCYCLE_CLS:
            continue
        other_imgs.update(imgs)

    other_imgs = list(other_imgs - selected_imgs)

    # 从其他类别中随机补足
    sampled_others = random.sample(
        other_imgs,
        min(len(other_imgs), remaining_quota)
    )

    selected_imgs.update(sampled_others)

print("最终抽取图片总数：", len(selected_imgs))

# ===================== Step 4：重新划分 train / val =====================

selected_imgs = list(selected_imgs)
random.shuffle(selected_imgs)

train_count = int(len(selected_imgs) * TRAIN_RATIO)

train_imgs = selected_imgs[:train_count]
val_imgs = selected_imgs[train_count:]

print("mini train 数量：", len(train_imgs))
print("mini val 数量：", len(val_imgs))

# ===================== Step 5：复制文件 =====================

def copy_files(img_list, dst_img_dir, dst_lbl_dir):
    """从原始 train / val 中找到图片并复制"""
    for img in img_list:
        lbl = img.replace(IMG_EXT, ".txt")

        for img_dir, lbl_dir in [
            (src_img_train, src_lbl_train),
            (src_img_val, src_lbl_val)
        ]:
            img_path = os.path.join(img_dir, img)
            lbl_path = os.path.join(lbl_dir, lbl)

            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(dst_img_dir, img))
                shutil.copy(lbl_path, os.path.join(dst_lbl_dir, lbl))
                break

# 拷贝 mini train
copy_files(train_imgs, dst_img_train, dst_lbl_train)

# 拷贝 mini val
copy_files(val_imgs, dst_img_val, dst_lbl_val)

print("✅ vehicle_orientation_mini 构建完成（总体 1/4，motorcycle 全保留）")
