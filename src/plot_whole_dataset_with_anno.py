from pathlib import Path
import yaml
import cv2
import numpy as np
from tqdm import tqdm

# =====================================================
# تنظیم مسیرها
# =====================================================

DATASET_DIR = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\train")

IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

YAML_PATH = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\data.yaml")

# پوشه خروجی
OUTPUT_DIR = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\train\class_visualization")

# کلاس‌هایی که پوشه دارند
TARGET_CLASSES = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]

# کلاس‌هایی که همیشه نمایش داده می‌شوند
ALWAYS_DRAW_CLASSES = [3, 9]

# =====================================================
# خواندن yaml
# =====================================================

with open(YAML_PATH, "r") as f:
    yaml_data = yaml.safe_load(f)

names_data = yaml_data["names"]

if isinstance(names_data, list):

    class_names = {
        idx: name
        for idx, name in enumerate(names_data)
    }

elif isinstance(names_data, dict):

    class_names = {
        int(k): v
        for k, v in names_data.items()
    }

else:
    raise ValueError("Unsupported names format in yaml")

# =====================================================
# ساخت پوشه‌ها
# =====================================================

for cls_id in TARGET_CLASSES:

    cls_name = class_names.get(
        cls_id,
        f"class_{cls_id}"
    )

    class_folder = OUTPUT_DIR / f"{cls_id}_{cls_name}"

    class_folder.mkdir(
        parents=True,
        exist_ok=True
    )

# =====================================================
# رنگ‌ها
# =====================================================

np.random.seed(42)

ALL_CLASSES = (
    TARGET_CLASSES +
    ALWAYS_DRAW_CLASSES
)

CLASS_COLORS = {}

for cls_id in ALL_CLASSES:

    CLASS_COLORS[cls_id] = tuple(
        int(x)
        for x in np.random.randint(0, 255, 3)
    )

# =====================================================
# فرمت تصاویر
# =====================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

# =====================================================
# پیدا کردن تصویر
# =====================================================

def find_image(stem):

    for ext in IMAGE_EXTENSIONS:

        p = IMAGES_DIR / f"{stem}{ext}"

        if p.exists():
            return p

    return None

# =====================================================
# رسم polygon
# =====================================================

def draw_polygon(
    image,
    cls_id,
    points,
):

    color = CLASS_COLORS[cls_id]

    points_np = np.array(
        points,
        dtype=np.int32
    )

    # رسم polygon
    cv2.polylines(
        image,
        [points_np],
        isClosed=True,
        color=color,
        thickness=1
    )

    # اسم کلاس
    cls_name = class_names.get(
        cls_id,
        f"class_{cls_id}"
    )

    x_text = points_np[0][0]
    y_text = points_np[0][1] - 10

    cv2.putText(
        image,
        cls_name,
        (x_text, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )

# =====================================================
# پردازش
# =====================================================

label_files = list(
    LABELS_DIR.glob("*.txt")
)

print(f"Found {len(label_files)} labels")

for label_path in tqdm(label_files):

    stem = label_path.stem

    image_path = find_image(stem)

    if image_path is None:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    # همه object ها
    parsed_objects = []

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 7:
            continue

        cls_id = int(parts[0])

        coords = list(
            map(float, parts[1:])
        )

        points = []

        for i in range(0, len(coords), 2):

            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)

            points.append([x, y])

        parsed_objects.append({
            "class_id": cls_id,
            "points": points,
        })

    # ============================================
    # برای هر کلاس target
    # ============================================

    present_target_classes = set([
        x["class_id"]
        for x in parsed_objects
        if x["class_id"] in TARGET_CLASSES
    ])

    for target_cls in present_target_classes:

        vis_img = image.copy()

        # ----------------------------------------
        # رسم کلاس target
        # ----------------------------------------

        for obj in parsed_objects:

            cls_id = obj["class_id"]

            # کلاس اصلی
            if cls_id == target_cls:

                draw_polygon(
                    vis_img,
                    cls_id,
                    obj["points"]
                )

            # cornea و pupil
            elif cls_id in ALWAYS_DRAW_CLASSES:

                draw_polygon(
                    vis_img,
                    cls_id,
                    obj["points"]
                )

        # ----------------------------------------
        # ذخیره
        # ----------------------------------------

        cls_name = class_names.get(
            target_cls,
            f"class_{target_cls}"
        )

        output_dir = (
            OUTPUT_DIR /
            f"{target_cls}_{cls_name}"
        )

        output_path = (
            output_dir /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            vis_img
        )

print("\n===================================")
print("Done.")
print(f"Saved to: {OUTPUT_DIR}")
print("===================================")