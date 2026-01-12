import os
import random
import shutil

# =======================
# 0. 固定随机种子（关键）
# =======================

SEED = 42
random.seed(SEED)

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

# 创建目标目录
for p in [DST_IMG_TRAIN, DST_IMG_VAL, DST_LAB_TRAIN, DST_LAB_VAL]:
    os.makedirs(p, exist_ok=True)

# =======================
# 2. 类别 ID 定义
# =======================

BUS_ID = 2
MOTOR_ID = 1

TRAIN_RATIO = 0.8

# =======================
# 3. 去重集合
# =======================

# 只要 label 中出现 bus 或 motorcycle → 必保留
must_keep = set()

# 其余图片
others = set()

# =======================
# 4. 合并 train + val，并逐个检查 label
# =======================

for img_dir, lab_dir in zip(SRC_IMG_DIRS, SRC_LAB_DIRS):

    # ⚠️ sorted：保证不同机器 / 系统遍历顺序一致
    for img_name in sorted(os.listdir(img_dir)):

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

        # 只要出现 bus 或 motorcycle，整张图直接进 must_keep
        if BUS_ID in cls_ids or MOTOR_ID in cls_ids:
            must_keep.add(item)
        else:
            others.add(item)

# =======================
# 5. 统计 & 目标数量
# =======================

total_original = len(must_keep) + len(others)
target_total = total_original // 4

print(f"原始总图片数: {total_original}")
print(f"目标 mini 数量 (/4): {target_total}")
print(f"必须保留(bus + motorcycle): {len(must_keep)}")

remaining = target_total - len(must_keep)

if remaining <= 0:
    raise RuntimeError(
        "❌ bus + motorcycle 图片数量已经超过原始数据集的 1/4，无法满足约束"
    )

print(f"需要从其余图片中随机抽取: {remaining}")

# =======================
# 6. 随机抽取其余图片
# =======================

sampled_others = random.sample(list(others), remaining)

selected = list(must_keep) + sampled_others

# =======================
# 7. 打乱并重新划分 train / val
# =======================

random.shuffle(selected)

train_num = int(len(selected) * TRAIN_RATIO)

train_items = selected[:train_num]
val_items   = selected[train_num:]

# =======================
# 8. 拷贝文件
# =======================

def copy_items(items, img_dst, lab_dst):
    for img_path, lab_path in items:
        shutil.copy(img_path, img_dst)
        shutil.copy(lab_path, lab_dst)

copy_items(train_items, DST_IMG_TRAIN, DST_LAB_TRAIN)
copy_items(val_items,   DST_IMG_VAL,   DST_LAB_VAL)

# =======================
# 9. 完成提示
# =======================

print("\n✅ vehicle_orientation_mini 构建完成")
print(f"Total: {len(selected)}（严格 = 原始 /4）")
print(f"Train: {len(train_items)}")
print(f"Val:   {len(val_items)}")
print(f"Random Seed: {SEED}")
