import os
from collections import defaultdict, Counter
import pandas as pd
import yaml

# =========================================================
# CONFIG
# =========================================================

# مسیر دیتاست
DATASET_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\new\txt\padded_splited_data_114_0"

# پوشه label های validation
VAL_LABEL_DIR = os.path.join(DATASET_DIR, "val", "labels")

# فایل yaml
YAML_PATH = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\new\txt\raw\data.yaml"

# =========================================================
# LOAD CLASS NAMES FROM YAML
# =========================================================

with open(YAML_PATH, "r") as f:
    data_yaml = yaml.safe_load(f)

# اگر names به صورت dict باشد
if isinstance(data_yaml["names"], dict):
    CLASS_NAMES = data_yaml["names"]

# اگر names به صورت list باشد
elif isinstance(data_yaml["names"], list):
    CLASS_NAMES = {
        i: name for i, name in enumerate(data_yaml["names"])
    }

else:
    raise ValueError("Unsupported YAML names format")

# =========================================================
# PATIENT ID EXTRACTION
# =========================================================

def extract_patient_id(filename):
    """
    Example:
    case5016_02_png.rf.xxxxx.txt
    -> case5016
    """

    base = os.path.basename(filename)

    patient_id = base.split("_")[0]

    return patient_id

# =========================================================
# COLLECT VALIDATION FILES
# =========================================================

label_files = [
    f for f in os.listdir(VAL_LABEL_DIR)
    if f.endswith(".txt")
]

print(f"\nTotal validation label files: {len(label_files)}")

# =========================================================
# PATIENT STATISTICS
# =========================================================

patient_stats = defaultdict(lambda: {
    "num_images": 0,
    "class_counter": Counter(),
    "files": []
})

# =========================================================
# PROCESS EACH LABEL FILE
# =========================================================

for label_file in label_files:

    patient_id = extract_patient_id(label_file)

    patient_stats[patient_id]["num_images"] += 1

    patient_stats[patient_id]["files"].append(label_file)

    full_label_path = os.path.join(
        VAL_LABEL_DIR,
        label_file
    )

    with open(full_label_path, "r") as lf:
        lines = lf.readlines()

    # فقط presence هر کلاس در هر تصویر
    present_classes = set()

    for line in lines:

        parts = line.strip().split()

        if len(parts) == 0:
            continue

        class_id = int(parts[0])

        present_classes.add(class_id)

    for cid in present_classes:
        patient_stats[patient_id]["class_counter"][cid] += 1

# =========================================================
# CREATE SUMMARY TABLE
# =========================================================

rows = []

# کلاس های rare
RARE_CLASSES = [
    "Cap Cystotome",
    "Cap Forceps",
    "Forceps",
    "Lens Injector",
    "Primary Knife",
    "Secondary Knife"
]

for patient_id, stats in patient_stats.items():

    row = {}

    row["patient_id"] = patient_id

    row["num_images"] = stats["num_images"]

    row["num_unique_classes"] = len(
        stats["class_counter"]
    )

    total_occ = sum(
        stats["class_counter"].values()
    )

    row["total_class_occurrences"] = total_occ

    # ذخیره تعداد هر کلاس
    for cid, cname in CLASS_NAMES.items():

        count = stats["class_counter"].get(cid, 0)

        row[cname] = count

    # rare classes present
    rare_present = []

    for cname in RARE_CLASSES:

        if row.get(cname, 0) > 0:
            rare_present.append(cname)

    row["rare_classes_present"] = ", ".join(
        rare_present
    )

    rows.append(row)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(rows)

# =========================================================
# SORT
# =========================================================

df = df.sort_values(
    by=[
        "num_unique_classes",
        "total_class_occurrences",
        "num_images"
    ],
    ascending=False
)

# =========================================================
# SAVE CSV
# =========================================================

output_csv = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\new\txt\padded_splited_data_114_0\validation_patient_statistics.csv"

df.to_csv(output_csv, index=False)

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n===================================================")
print("VALIDATION PATIENT STATISTICS")
print("===================================================")

print(df.head(20))

print(f"\nSaved CSV -> {output_csv}")