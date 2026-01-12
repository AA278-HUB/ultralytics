import os
import random
import shutil
from collections import defaultdict

# ================= 配置 =================

SRC_ROOT = r"E:\datasets\vehicle_orientation"
DST_ROOT = r"E:\datasets\vehicle_orientation_mini"

IMG_EXT = ".jpg"
TRAIN_RATIO = 0.8
SEED = 42
random.seed(SEED)

# 类别定义
NAMES = ['car', 'motorcycle', 'bus', 'truck']
CAR_CLS = 0

# ================= 工具 =================

def mkdir(p):
    os.makedirs(p, exist_ok=True)

# ================= 路径 =================

src_img_train = os.path.join(SRC_ROOT, "images", "train")
src_lbl_train = os.path.join(SRC_ROOT, "labels", "train")
src_img_val   = os.path.join(SRC_ROOT, "images", "val")
src_lbl_val   = os.path.join(SRC_ROOT, "labels", "val")

dst_img_train = os.path.join(DST_ROOT, "images", "train")
dst_lbl_train = os.path.join(DST_ROOT, "labels", "train")
dst_img_val   = os.path.join(DST_ROOT, "images", "val")
dst_lbl_val   = os.path.join(DST_ROOT, "labels", "val")

for d in [dst_img_train, dst_lbl_train, dst_img_val, dst_lbl_val]:
    mkdir(d)

# ================= 统计 train + val =================

class_to_imgs = defaultdict(set)
all_images = set()

def collect(label_dir):
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        img = f.replace(".txt", IMG_EXT)
        all_images.add(img)
        with open(os.path.join(label_dir, f)) as fp:
            for line in fp:
                cls = int(line.split()[0])
                class_to_imgs[cls].add(img)

collect(src_lbl_train)
collect(src_lbl_val)

print("===== 原始类别统计 =====")
for k in sorted(class_to_imgs):
    print(NAMES[k], len(class_to_imgs[k]))

# ================= 均衡策略 =================

# 非 car 类别的最大数量
non_car_counts = [
    len(class_to_imgs[c]) for c in class_to_imgs if c != CAR_CLS
]
target_per_class = max(non_car_counts)

print("目标每类数量（car 会被压到这个量级）:", target_per_class)

selected_imgs = set()

for cls, imgs in class_to_imgs.items():
    imgs = list(imgs)
    if cls == CAR_CLS:
        sampled = random.sample(imgs, min(len(imgs), target_per_class))
    else:
        sampled = random.sample(imgs, min(len(imgs), target_per_class))
    selected_imgs.update(sampled)

print("最终图片总数:", len(selected_imgs))

# ================= 重新划分 train / val =================

selected_imgs = list(selected_imgs)
random.shuffle(selected_imgs)

train_n = int(len(selected_imgs) * TRAIN_RATIO)
train_imgs = selected_imgs[:train_n]
val_imgs   = selected_imgs[train_n:]

# ================= 复制文件 =================

def copy_imgs(img_list, dst_img, dst_lbl):
    for img in img_list:
        lbl = img.replace(IMG_EXT, ".txt")
        for idir, ldir in [
            (src_img_train, src_lbl_train),
            (src_img_val, src_lbl_val)
        ]:
            ip = os.path.join(idir, img)
            lp = os.path.join(ldir, lbl)
            if os.path.exists(ip):
                shutil.copy(ip, os.path.join(dst_img, img))
                shutil.copy(lp, os.path.join(dst_lbl, lbl))
                break

copy_imgs(train_imgs, dst_img_train, dst_lbl_train)
copy_imgs(val_imgs, dst_img_val, dst_lbl_val)

print("===== mini 数据集完成 =====")
print("train:", len(train_imgs))
print("val:", len(val_imgs))
