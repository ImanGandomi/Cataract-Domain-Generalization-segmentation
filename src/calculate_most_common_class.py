from pathlib import Path
from collections import defaultdict
import yaml

labels_dir = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\raw\train\labels")
yaml_path = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\raw\data.yaml")

# خواندن نام کلاس‌ها
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

class_names = data["names"]

# video_name -> set(classes present in video)
video_classes = defaultdict(set)

for txt_file in labels_dir.glob("*.txt"):

    base_name = txt_file.stem.split(".rf.")[0]
    parts = base_name.split("_")

    if len(parts) < 6:
        continue

    # SE_{video_name}_0003_S1_1568_jpg
    video_name = "_".join(parts[1:-4])

    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])
            video_classes[video_name].add(class_id)

# شمارش تعداد ویدیوهای حاوی هر کلاس
class_video_count = defaultdict(int)

for classes_in_video in video_classes.values():
    for cls in classes_in_video:
        class_video_count[cls] += 1

# مرتب‌سازی نزولی
ranking = sorted(
    class_video_count.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\n=== Classes Ranked By Number Of Videos ===\n")

for rank, (cls_id, num_videos) in enumerate(ranking, start=1):

    cls_name = (
        class_names[cls_id]
        if cls_id < len(class_names)
        else f"class_{cls_id}"
    )

    print(
        f"{rank:2d}. {cls_name:<25} "
        f"({num_videos} videos)"
    )