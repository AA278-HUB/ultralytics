# ================= 引入库 =================
import os                  # 处理路径、文件夹
import random              # 随机抽样
import shutil              # 复制文件
from collections import defaultdict  # 自动创建字典

# ================= 配置区（你主要看这里） =================

# 原始数据集
SRC_ROOT = r"E:\datasets\vehicle_orientation"

# 新数据集
DST_ROOT = r"E:\datasets\vehicle_orientation_mini"

# 图片后缀
IMG_EXT = ".jpg"   # 如果你是 png，改成 ".png"

# 总体只保留 1/4
SAMPLE_RATIO = 0.25

# motorcycle 在 names 里的 id
MOTORCYCLE_CLS = 1

# mini 数据集 train / val 比例
TRAIN_RATIO = 0.8

# 固定随机数，保证可复现
SEED = 42
random.seed(SEED)

# ================= 工具函数 =================

def mkdir(path):
    # 如果文件夹不存在就创建
    os.makedirs(path, exist_ok=True)

# ================= 原始路径 =================

src_img_train = os.path.join(SRC_ROOT, "images", "train")
src_lbl_train = os.path.join(SRC_ROOT, "labels", "train")
src_img_val   = os.path.join(SRC_ROOT, "images", "val")
src_lbl_val   = os.path.join(SRC_ROOT, "labels", "val")

# ================= mini 路径 =================

dst_img_train = os.path.join(DST_ROOT, "images", "train")
dst_lbl_train = os.path.join(DST_ROOT, "labels", "train")
dst_img_val   = os.path.join(DST_ROOT, "images", "val")
dst_lbl_val   = os.path.join(DST_ROOT, "labels", "val")

# 创建文件夹
for d in [dst_img_train, dst_lbl_train, dst_img_val, dst_lbl_val]:
    mkdir(d)

# ================= Step 1：合并 train + val =================

# class_id -> 图片集合
class_to_imgs = defaultdict(set)

# 所有图片的集合（不区分 train / val）
all_images = set()

def collect_from(label_dir):
    # 遍历某个 labels 目录
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

# ================= Step 2：计算每类目标数量 =================

motor_imgs = class_to_imgs[MOTORCYCLE_CLS]

target_per_class = int(len(motor_imgs) * SAMPLE_RATIO)

print("motorcycle 总图片数：", len(motor_imgs))
print("每个类别目标数量：", target_per_class)

# ================= Step 3：按类别均衡抽取 =================

selected_imgs = set()

for cls, imgs in class_to_imgs.items():
    imgs = list(imgs)
    n = min(len(imgs), target_per_class)
    sampled = random.sample(imgs, n)
    selected_imgs.update(sampled)

print("抽取后的总图片数：", len(selected_imgs))

# ================= Step 4：重新划分 train / val =================

selected_imgs = list(selected_imgs)
random.shuffle(selected_imgs)

train_count = int(len(selected_imgs) * TRAIN_RATIO)

train_imgs = selected_imgs[:train_count]
val_imgs   = selected_imgs[train_count:]

# ================= Step 5：拷贝文件 =================

def copy_files(img_list, dst_img_dir, dst_lbl_dir):
    for img in img_list:
        lbl = img.replace(IMG_EXT, ".txt")

        # 原始图片可能在 train 或 val
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

# 拷贝 train
copy_files(train_imgs, dst_img_train, dst_lbl_train)

# 拷贝 val
copy_files(val_imgs, dst_img_val, dst_lbl_val)

print("✅ vehicle_orientation_mini 构建完成（先合并，再抽样，再划分）")
