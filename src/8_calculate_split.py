from pathlib import Path
import yaml
from collections import defaultdict

# ==================================================
# مسیر دیتاست
# ==================================================
dataset_path = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\padded_splited_data")

# فایل yaml
yaml_path = dataset_path / r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\raw\data.yaml"

# ==================================================
# خواندن کلاس‌ها
# ==================================================
with open(yaml_path, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

# names ممکنه list یا dict باشه
if isinstance(data_yaml["names"], list):
    class_names = {i: name for i, name in enumerate(data_yaml["names"])}
else:
    class_names = {int(k): v for k, v in data_yaml["names"].items()}

# ==================================================
# split ها
# ==================================================
splits = ["train", "val", "test"]

# ==================================================
# شمارش فایل‌های label
# ==================================================
label_file_counts = {}

for split in splits:

    labels_path = dataset_path / split / "labels"

    label_files = list(labels_path.glob("*.txt"))

    label_file_counts[split] = len(label_files)

total_label_files = sum(label_file_counts.values())

# ==================================================
# نمایش توزیع split ها
# ==================================================
print("\n" + "="*60)
print("DATASET SPLIT DISTRIBUTION (LABEL FILES)")
print("="*60)

print(f"\nTotal Label Files: {total_label_files}\n")

for split in splits:

    count = label_file_counts[split]

    percentage = (
        (count / total_label_files) * 100
        if total_label_files > 0 else 0
    )

    print(
        f"{split:<5}: "
        f"{count:>5} files "
        f"({percentage:6.2f}%)"
    )

# ==================================================
# شمارش instance های هر کلاس
# ==================================================
class_counts = {
    split: defaultdict(int)
    for split in splits
}

total_class_counts = defaultdict(int)

# ==================================================
# خواندن label ها
# فرمت YOLO Segmentation:
# class x1 y1 x2 y2 ...
# ==================================================
for split in splits:

    labels_path = dataset_path / split / "labels"

    label_files = list(labels_path.glob("*.txt"))

    for label_file in label_files:

        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])

            # هر line = یک instance
            class_counts[split][class_id] += 1
            total_class_counts[class_id] += 1

# ==================================================
# نمایش توزیع کلاس‌ها
# ==================================================
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)

for class_id, class_name in class_names.items():

    total = total_class_counts[class_id]

    print(f"\nClass {class_id} -> {class_name}")
    print(f"TOTAL INSTANCES: {total}")

    if total == 0:
        print("No samples found.")
        continue

    for split in splits:

        count = class_counts[split][class_id]

        percentage = (count / total) * 100

        print(
            f"{split:<5}: "
            f"{count:>5} instances "
            f"({percentage:6.2f}%)"
        )