import os
import random
import shutil
from collections import defaultdict

# =======================
# 1. 路径配置
# =======================

SRC_ROOT = r"E:\datasets\vehicle_orientation"
DST_ROOT = r"E:\datasets\vehicle_orientation_mini"

SRC_IMG = [
    os.path.join(SRC_ROOT, "images", "train"),
    os.path.join(SRC_ROOT, "images", "val"),
]
SRC_LAB = [
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
# 2. 类别定义
# =======================

ID2NAME = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}

KEEP_ALL = {"motorcycle", "bus"}
SAMPLE_CLASSES = {"car", "truck"}

TRAIN_RATIO = 0.8

# =======================
# 3. 合并 train + val
# =======================

class_to_items = defaultdict(list)

for img_dir, lab_dir in zip(SRC_IMG, SRC_LAB):
    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_name)
        lab_path = os.path.join(lab_dir, img_name.replace(".jpg", ".txt"))

        if not os.path.exists(lab_path):
            continue

        with open(lab_path, "r") as f:
            lines = f.readlines()
            if not lines:
                continue

            cls_id = int(lines[0].split()[0])
            cls_name = ID2NAME[cls_id]

        class_to_items[cls_name].append((img_path, lab_path))

# =======================
# 4. 原始统计
# =======================

print("📊 原始数据集统计：")
total_count = 0
for cls, items in class_to_items.items():
    print(f"{cls}: {len(items)}")
    total_count += len(items)

target_total = total_count // 4
print(f"\n🎯 目标 mini 总数：{target_total} (原始 /4)")

# =======================
# 5. 全保留的小类
# =======================

selected = []

keep_count = 0
for cls in KEEP_ALL:
    items = class_to_items.get(cls, [])
    selected.extend(items)
    keep_count += len(items)
    print(f"✅ {cls} 全保留：{len(items)}")

# =======================
# 6. 给 car + truck 分配剩余额度
# =======================

remaining_quota = target_total - keep_count
assert remaining_quota > 0, "❌ motorcycle + bus 已超过 1/4，总量不可能满足"

big_items = []
for cls in SAMPLE_CLASSES:
    big_items.extend(class_to_items.get(cls, []))

big_total = len(big_items)

print(f"\n📌 car + truck 总数：{big_total}")
print(f"📌 剩余可用名额：{remaining_quota}")

# 按比例抽取
num_to_sample = remaining_quota
sampled_big = random.sample(big_items, num_to_sample)

selected.extend(sampled_big)

print(f"✂️ car + truck 实际抽取：{len(sampled_big)}")

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

def copy(items, img_dst, lab_dst):
    for img, lab in items:
        shutil.copy(img, img_dst)
        shutil.copy(lab, lab_dst)

copy(train_items, DST_IMG_TRAIN, DST_LAB_TRAIN)
copy(val_items,   DST_IMG_VAL,   DST_LAB_VAL)

# =======================
# 9. 最终结果
# =======================

print("\n✅ vehicle_orientation_mini 构建完成")
print(f"Total: {len(selected)}（严格 = 原始 /4）")
print(f"Train: {len(train_items)}")
print(f"Val:   {len(val_items)}")
