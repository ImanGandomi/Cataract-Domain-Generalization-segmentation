# ########################################################################################### برای بررسی کیفیت حاشیه نویسی پیوپیل و کورنیا
# from pathlib import Path
# import numpy as np
# import cv2
# from collections import defaultdict

# # =========================================
# # تنظیمات
# # =========================================

# labels_dir = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt\labels")
# images_dir = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt\images")

# # کلاس‌ها
# CORNEA_CLASS = 3
# PUPIL_CLASS = 9

# # حداقل تعداد نقاط polygon
# MIN_POINTS = 12

# # threshold برای smooth بودن
# # هرچه کمتر → سخت‌گیرتر
# SMOOTHNESS_THRESHOLD = 0.55

# # =========================================
# # توابع کمکی
# # =========================================

# def polygon_area(points):
#     """
#     محاسبه مساحت polygon
#     """
#     pts = np.array(points, dtype=np.float32)

#     x = pts[:, 0]
#     y = pts[:, 1]

#     return 0.5 * abs(
#         np.dot(x, np.roll(y, 1)) -
#         np.dot(y, np.roll(x, 1))
#     )


# def polygon_smoothness(points):
#     """
#     بررسی smooth بودن polygon

#     ایده:
#     نسبت محیط polygon به محیط convex hull
#     """

#     pts = np.array(points, dtype=np.float32)

#     if len(pts) < 3:
#         return 0

#     contour = pts.reshape((-1, 1, 2)).astype(np.float32)

#     perimeter = cv2.arcLength(contour, True)

#     hull = cv2.convexHull(contour)
#     hull_perimeter = cv2.arcLength(hull, True)

#     if hull_perimeter == 0:
#         return 0

#     ratio = hull_perimeter / perimeter

#     return ratio


# def parse_yolo_segmentation(label_path):

#     objects = []

#     with open(label_path, "r") as f:
#         lines = f.readlines()

#     for line in lines:

#         parts = line.strip().split()

#         if len(parts) < 7:
#             continue

#         cls_id = int(parts[0])

#         coords = list(map(float, parts[1:]))

#         points = []

#         for i in range(0, len(coords), 2):
#             points.append([
#                 coords[i],
#                 coords[i + 1]
#             ])

#         objects.append({
#             "class_id": cls_id,
#             "points": points,
#         })

#     return objects


# # =========================================
# # پردازش دیتاست
# # =========================================

# issues = defaultdict(list)

# label_files = list(labels_dir.glob("*.txt"))

# print(f"Found {len(label_files)} labels\n")

# for label_path in label_files:

#     image_name = label_path.stem

#     try:

#         objects = parse_yolo_segmentation(label_path)

#         corneas = [
#             x for x in objects
#             if x["class_id"] == CORNEA_CLASS
#         ]

#         pupils = [
#             x for x in objects
#             if x["class_id"] == PUPIL_CLASS
#         ]

#         # =====================================
#         # بررسی وجود کلاس‌ها
#         # =====================================

#         if len(corneas) == 0:
#             issues["missing_cornea"].append(image_name)

#         if len(pupils) == 0:
#             issues["missing_pupil"].append(image_name)

#         if len(corneas) == 0 or len(pupils) == 0:
#             continue

#         # فرض:
#         # بزرگ‌ترین object هر کلاس مهم است

#         cornea = max(
#             corneas,
#             key=lambda x: polygon_area(x["points"])
#         )

#         pupil = max(
#             pupils,
#             key=lambda x: polygon_area(x["points"])
#         )

#         cornea_area = polygon_area(cornea["points"])
#         pupil_area = polygon_area(pupil["points"])

#         # =====================================
#         # بررسی مساحت
#         # =====================================

#         if cornea_area <= pupil_area:
#             issues["cornea_smaller_than_pupil"].append(
#                 image_name
#             )

#         # =====================================
#         # بررسی تعداد نقاط polygon
#         # =====================================

#         if len(cornea["points"]) < MIN_POINTS:
#             issues["cornea_low_polygon_points"].append(
#                 image_name
#             )

#         if len(pupil["points"]) < MIN_POINTS:
#             issues["pupil_low_polygon_points"].append(
#                 image_name
#             )

#         # =====================================
#         # بررسی smooth بودن
#         # =====================================

#         cornea_smoothness = polygon_smoothness(
#             cornea["points"]
#         )

#         pupil_smoothness = polygon_smoothness(
#             pupil["points"]
#         )

#         if cornea_smoothness < SMOOTHNESS_THRESHOLD:
#             issues["cornea_not_smooth"].append(
#                 image_name
#             )

#         if pupil_smoothness < SMOOTHNESS_THRESHOLD:
#             issues["pupil_not_smooth"].append(
#                 image_name
#             )

#     except Exception as e:
#         issues["parse_errors"].append(
#             f"{image_name} --> {e}"
#         )

# # =========================================
# # گزارش نهایی
# # =========================================

# print("\n==============================")
# print("DATASET QUALITY REPORT")
# print("==============================\n")

# for category, files in issues.items():

#     print(f"\n[{category}]")
#     print(f"Count: {len(files)}")

#     for f in files[:30]:
#         print(" -", f)

#     if len(files) > 30:
#         print(f" ... and {len(files)-30} more")

# # =========================================
# # ذخیره گزارش
# # =========================================

# report_path = Path("dataset_quality_report.txt")

# with open(report_path, "w") as f:

#     for category, files in issues.items():

#         f.write(f"\n[{category}]\n")
#         f.write(f"Count: {len(files)}\n")

#         for item in files:
#             f.write(f"{item}\n")

# print("\n==============================")
# print(f"Report saved to: {report_path}")
# print("==============================")

####################################################################################################### برای مقایسه دو نسخه از یه دیتاست
from pathlib import Path
import hashlib
import numpy as np
import cv2
from collections import defaultdict

# =====================================================
# تنظیم مسیر دیتاست‌ها
# =====================================================

OLD_IMAGES_DIR = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt\images")
OLD_LABELS_DIR = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt\labels")

NEW_IMAGES_DIR = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\6094_Cataract_Seg_Dataset_Final2\train\images")
NEW_LABELS_DIR = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\6094_Cataract_Seg_Dataset_Final2\train\labels")

# =====================================================
# تنظیمات
# =====================================================

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# tolerance برای تفاوت float ها
COORD_TOLERANCE = 1e-5

# =====================================================
# توابع کمکی
# =====================================================

def file_md5(path):
    """
    هش فایل برای مقایسه دقیق
    """

    md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)

    return md5.hexdigest()


def find_image(images_dir, stem):

    for ext in IMAGE_EXTENSIONS:

        p = images_dir / f"{stem}{ext}"

        if p.exists():
            return p

    return None


def parse_yolo_segmentation(label_path):

    objects = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 7:
            continue

        cls_id = int(parts[0])

        coords = list(map(float, parts[1:]))

        points = []

        for i in range(0, len(coords), 2):
            points.append([
                coords[i],
                coords[i + 1]
            ])

        objects.append({
            "class_id": cls_id,
            "points": points,
            "raw": line.strip(),
        })

    return objects


def compare_polygons(poly1, poly2):

    if poly1["class_id"] != poly2["class_id"]:
        return False

    pts1 = poly1["points"]
    pts2 = poly2["points"]

    if len(pts1) != len(pts2):
        return False

    for p1, p2 in zip(pts1, pts2):

        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])

        if dx > COORD_TOLERANCE:
            return False

        if dy > COORD_TOLERANCE:
            return False

    return True


# =====================================================
# جمع‌آوری فایل‌ها
# =====================================================

old_label_files = {
    x.stem: x
    for x in OLD_LABELS_DIR.glob("*.txt")
}

new_label_files = {
    x.stem: x
    for x in NEW_LABELS_DIR.glob("*.txt")
}

all_names = sorted(
    set(old_label_files.keys()) |
    set(new_label_files.keys())
)

print(f"Total samples to compare: {len(all_names)}")

# =====================================================
# گزارش اختلاف‌ها
# =====================================================

report = defaultdict(list)

# =====================================================
# مقایسه
# =====================================================

for name in all_names:

    old_label = old_label_files.get(name)
    new_label = new_label_files.get(name)

    old_img = find_image(OLD_IMAGES_DIR, name)
    new_img = find_image(NEW_IMAGES_DIR, name)

    # =================================================
    # وجود فایل‌ها
    # =================================================

    if old_label is None:
        report["missing_label_in_old"].append(name)

    if new_label is None:
        report["missing_label_in_new"].append(name)

    if old_img is None:
        report["missing_image_in_old"].append(name)

    if new_img is None:
        report["missing_image_in_new"].append(name)

    if (
        old_label is None or
        new_label is None or
        old_img is None or
        new_img is None
    ):
        continue

    # =================================================
    # مقایسه تصویر
    # =================================================

    try:

        old_img_hash = file_md5(old_img)
        new_img_hash = file_md5(new_img)

        if old_img_hash != new_img_hash:
            report["image_changed"].append(name)

    except Exception as e:
        report["image_compare_error"].append(
            f"{name} --> {e}"
        )

    # =================================================
    # مقایسه label
    # =================================================

    try:

        old_objects = parse_yolo_segmentation(old_label)
        new_objects = parse_yolo_segmentation(new_label)

        # تعداد object ها
        if len(old_objects) != len(new_objects):

            report["different_object_count"].append(name)
            continue

        # مقایسه polygon ها
        mismatch_found = False

        for old_obj, new_obj in zip(
            old_objects,
            new_objects
        ):

            # کلاس متفاوت
            if old_obj["class_id"] != new_obj["class_id"]:

                report["different_class_ids"].append(name)
                mismatch_found = True
                break

            # تعداد نقاط متفاوت
            if len(old_obj["points"]) != len(new_obj["points"]):

                report["different_polygon_point_count"].append(
                    name
                )

                mismatch_found = True
                break

            # مختصات متفاوت
            if not compare_polygons(old_obj, new_obj):

                report["different_polygon_coordinates"].append(
                    name
                )

                mismatch_found = True
                break

        if mismatch_found:
            continue

    except Exception as e:

        report["label_compare_error"].append(
            f"{name} --> {e}"
        )

# =====================================================
# چاپ گزارش
# =====================================================

print("\n======================================")
print("DATASET COMPARISON REPORT")
print("======================================")

for category, items in report.items():

    print(f"\n[{category}]")
    print(f"Count: {len(items)}")

    for item in items[:40]:
        print(" -", item)

    if len(items) > 40:
        print(f" ... and {len(items)-40} more")

# =====================================================
# ذخیره گزارش
# =====================================================

output_report = Path("dataset_diff_report.txt")

with open(output_report, "w") as f:

    f.write("DATASET COMPARISON REPORT\n")
    f.write("=====================================\n")

    for category, items in report.items():

        f.write(f"\n[{category}]\n")
        f.write(f"Count: {len(items)}\n")

        for item in items:
            f.write(f"{item}\n")

print("\n======================================")
print(f"Report saved to: {output_report}")
print("======================================")