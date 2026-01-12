import os
import random
import shutil

# =======================
# 1. 路径配置
# =======================

SRC_ROOT = r"E:\datasets\vehicle_orientation"
DST_ROOT = r"E:\datasets\vehicle_orientation_mini"

SRC_IMG_DIRS = [
    os.path.join(SRC_ROOT, "images", "train"),
    os.path.join(SRC_ROOT, "images", "val"),
]

SRC_LAB_DIRS = [
    os.path.join(SRC_ROOT, "labels", "train"),
    os.path.join(SRC_ROOT, "labels", "val"),
]

DST_IMG_TRAIN = os.path.join(DST_ROOT, "images", "train")
DST_IMG_VAL   = os.path.join(DST_ROOT, "images", "val")
DST_LAB_TRAIN = os.path.join(DST_ROOT, "labels", "train")
DST_LAB_VAL   = os.path.join(DST_ROOT, "labels", "val")

for p in [DST_IMG_TRAIN, DST_IMG_VAL, DST_LAB_TRAIN, DST_LAB_VAL]:
    os.makedirs(p, exist_ok=True)

# =======================
# 2. 类别 ID 定义
# =======================

BUS_ID = 2
MOTOR_ID = 1

TRAIN_RATIO = 0.8

# =======================
# 3. 两个“去重集合”
# =======================

must_keep = set()   # 只要出现 bus 或 motorcycle
others = set()      # 完全不含 bus / motorcycle

# =======================
# 4. 合并 train + val，并逐个检查 label
# =======================

for img_dir, lab_dir in zip(SRC_IMG_DIRS, SRC_LAB_DIRS):
    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_name)
        lab_path = os.path.join(lab_dir, img_name.replace(".jpg", ".txt"))

        if not os.path.exists(lab_path):
            continue

        # 读取 label 中所有目标的类别 id
        with open(lab_path, "r") as f:
            cls_ids = {int(line.split()[0]) for line in f if line.strip()}

        item = (img_path, lab_path)

        # 只要出现 bus 或 motorcycle，整张图强制保留
        if BUS_ID in cls_ids or MOTOR_ID in cls_ids:
            must_keep.add(item)
        else:
            others.add(item)

# =======================
# 5. 统计
# =======================

total_original = len(must_keep) + len(others)
target_total = total_original // 4

print(f"原始总图片数: {total_original}")
print(f"目标 mini 数量 (/4): {target_total}")
print(f"必须保留(bus+motor): {len(must_keep)}")

remaining = target_total - len(must_keep)
assert remaining > 0, "❌ bus + motorcycle 数量已经超过 /4，逻辑不成立"

print(f"需要从其他类别抽取: {remaining}")

# =======================
# 6. 随机抽取其余图片
# =======================

sampled_others = random.sample(list(others), remaining)

selected = list(must_keep) + sampled_others

# =======================
# 7. 打乱并划分 train / val
# =======================

random.shuffle(selected)

train_num = int(len(selected) * TRAIN_RATIO)

train_items = selected[:train_num]
val_items   = selected[train_num:]

# =======================
# 8. 拷贝文件
# =======================

def copy(items, img_dst, lab_dst):
    for img, lab in items:
        shutil.copy(img, img_dst)
        shutil.copy(lab, lab_dst)

copy(train_items, DST_IMG_TRAIN, DST_LAB_TRAIN)
copy(val_items,   DST_IMG_VAL,   DST_LAB_VAL)

# =======================
# 9. 完成提示
# =======================

print("\n✅ vehicle_orientation_mini 构建完成")
print(f"Total: {len(selected)}（严格 = 原始 /4）")
print(f"Train: {len(train_items)}")
print(f"Val:   {len(val_items)}")
