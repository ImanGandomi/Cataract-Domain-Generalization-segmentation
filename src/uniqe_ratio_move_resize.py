from pathlib import Path
from collections import Counter
from PIL import Image

# مسیر پوشه تصاویر
# فرمت‌ها را در صورت نیاز تغییر بده
image_dir = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\coco\raw\train")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# شمارش ابعاد یونیک
size_counter = Counter()

# لیست همه تصاویر
image_files = [
    f for f in image_dir.rglob("*")
    if f.suffix.lower() in image_extensions
]

print(f"Found {len(image_files)} images\n")

for img_path in image_files:
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            size_counter[(width, height)] += 1
    except Exception as e:
        print(f"Error reading {img_path}: {e}")

# نمایش نتایج
print("Unique image sizes:\n")

for (w, h), count in sorted(size_counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{w}x{h}  -->  {count} images")

############################################################################################### move
# from pathlib import Path
# from PIL import Image
# import shutil

# # =========================
# # تنظیم مسیرها
# # =========================

# # پوشه اصلی تصاویر
# images_dir = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt_Copy\images")

# # پوشه اصلی لیبل‌ها
# labels_dir = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt_Copy\labels")

# # پوشه خروجی
# output_root = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt_Copy")

# # فرمت‌های تصویری
# image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# # ابعاد هدف
# target_sizes = [
#     (720, 540),
#     (720, 528),
# ]

# # =========================
# # پردازش
# # =========================

# moved_count = 0

# for img_path in images_dir.rglob("*"):

#     if img_path.suffix.lower() not in image_extensions:
#         continue

#     try:
#         with Image.open(img_path) as img:
#             width, height = img.size

#         if (width, height) not in target_sizes:
#             continue

#         size_name = f"{width}x{height}"

#         # ساخت پوشه‌ها
#         target_images_dir = output_root / size_name / "images"
#         target_labels_dir = output_root / size_name / "labels"

#         target_images_dir.mkdir(parents=True, exist_ok=True)
#         target_labels_dir.mkdir(parents=True, exist_ok=True)

#         # مسیر لیبل متناظر
#         label_path = labels_dir / (img_path.stem + ".txt")

#         # مقصد فایل‌ها
#         target_img_path = target_images_dir / img_path.name
#         target_label_path = target_labels_dir / label_path.name

#         # =========================
#         # جابجایی تصویر
#         # =========================
#         shutil.move(str(img_path), str(target_img_path))

#         # =========================
#         # جابجایی لیبل (اگر وجود داشت)
#         # =========================
#         if label_path.exists():
#             shutil.move(str(label_path), str(target_label_path))
#         else:
#             print(f"[WARNING] Label not found for: {img_path.name}")

#         moved_count += 1
#         print(f"[MOVED] {img_path.name} --> {size_name}")

#     except Exception as e:
#         print(f"[ERROR] {img_path}: {e}")

# print("\n===================")
# print(f"Done. Moved {moved_count} images.")
# print("===================")



############################################################################################### resize
# from pathlib import Path
# from PIL import Image
# import shutil
# import random
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # =====================================
# # تنظیمات
# # =====================================

# input_dirs = [
#     Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt_Copy\720x540"),
#     Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt_Copy\720x528"),
# ]

# output_root = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\txt\raw\yolo_txt_Copy\resized_720x480")

# TARGET_W = 720
# TARGET_H = 480

# image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# # تعداد نمونه برای نمایش
# NUM_VIS_SAMPLES = 3


# # =====================================
# # رسم segmentation روی تصویر
# # =====================================

# def draw_yolo_segmentation(image, label_path):

#     img = image.copy()
#     h, w = img.shape[:2]

#     if not label_path.exists():
#         return img

#     with open(label_path, "r") as f:
#         lines = f.readlines()

#     for line in lines:
#         parts = line.strip().split()

#         if len(parts) < 7:
#             continue

#         cls_id = parts[0]
#         coords = list(map(float, parts[1:]))

#         points = []

#         for i in range(0, len(coords), 2):
#             x = int(coords[i] * w)
#             y = int(coords[i + 1] * h)
#             points.append([x, y])

#         points = np.array(points, dtype=np.int32)

#         cv2.polylines(
#             img,
#             [points],
#             isClosed=True,
#             color=(0, 255, 0),
#             thickness=2,
#         )

#     return img


# # =====================================
# # تبدیل مختصات segmentation
# # =====================================

# def resize_yolo_segmentation(
#     label_path,
#     output_label_path,
#     old_w,
#     old_h,
#     new_w,
#     new_h,
# ):

#     scale_x = new_w / old_w
#     scale_y = new_h / old_h

#     new_lines = []

#     with open(label_path, "r") as f:
#         lines = f.readlines()

#     for line in lines:

#         parts = line.strip().split()

#         if len(parts) < 7:
#             continue

#         cls_id = parts[0]
#         coords = list(map(float, parts[1:]))

#         updated_coords = []

#         for i in range(0, len(coords), 2):

#             x_norm = coords[i]
#             y_norm = coords[i + 1]

#             # تبدیل به pixel
#             x_pixel = x_norm * old_w
#             y_pixel = y_norm * old_h

#             # resize
#             x_pixel_new = x_pixel * scale_x
#             y_pixel_new = y_pixel * scale_y

#             # normalize جدید
#             x_new_norm = x_pixel_new / new_w
#             y_new_norm = y_pixel_new / new_h

#             updated_coords.extend([
#                 round(x_new_norm, 6),
#                 round(y_new_norm, 6)
#             ])

#         new_line = cls_id + " " + " ".join(map(str, updated_coords))
#         new_lines.append(new_line)

#     with open(output_label_path, "w") as f:
#         f.write("\n".join(new_lines))


# # =====================================
# # پردازش اصلی
# # =====================================

# visualization_samples = []

# for dataset_dir in input_dirs:

#     images_dir = dataset_dir / "images"
#     labels_dir = dataset_dir / "labels"

#     output_images_dir = output_root / dataset_dir.name / "images"
#     output_labels_dir = output_root / dataset_dir.name / "labels"

#     output_images_dir.mkdir(parents=True, exist_ok=True)
#     output_labels_dir.mkdir(parents=True, exist_ok=True)

#     image_files = [
#         x for x in images_dir.iterdir()
#         if x.suffix.lower() in image_extensions
#     ]

#     random.shuffle(image_files)

#     for idx, img_path in enumerate(image_files):

#         label_path = labels_dir / (img_path.stem + ".txt")

#         output_img_path = output_images_dir / img_path.name
#         output_label_path = output_labels_dir / label_path.name

#         try:

#             # خواندن تصویر
#             img_pil = Image.open(img_path)
#             old_w, old_h = img_pil.size

#             # resize تصویر
#             resized_img = img_pil.resize(
#                 (TARGET_W, TARGET_H),
#                 Image.BILINEAR
#             )

#             # ذخیره تصویر جدید
#             resized_img.save(output_img_path)

#             # resize label
#             if label_path.exists():
#                 resize_yolo_segmentation(
#                     label_path,
#                     output_label_path,
#                     old_w,
#                     old_h,
#                     TARGET_W,
#                     TARGET_H,
#                 )

#             # ذخیره نمونه برای visualization
#             if len(visualization_samples) < NUM_VIS_SAMPLES * len(input_dirs):
#                 visualization_samples.append({
#                     "before_img": img_path,
#                     "before_label": label_path,
#                     "after_img": output_img_path,
#                     "after_label": output_label_path,
#                     "source_size": dataset_dir.name,
#                 })

#             print(f"[DONE] {img_path.name}")

#         except Exception as e:
#             print(f"[ERROR] {img_path.name}: {e}")


# # =====================================
# # Visualization
# # =====================================

# print("\nShowing random samples...\n")

# for sample in visualization_samples:

#     before_img = cv2.imread(str(sample["before_img"]))
#     before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)

#     after_img = cv2.imread(str(sample["after_img"]))
#     after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)

#     before_vis = draw_yolo_segmentation(
#         before_img,
#         sample["before_label"]
#     )

#     after_vis = draw_yolo_segmentation(
#         after_img,
#         sample["after_label"]
#     )

#     plt.figure(figsize=(14, 7))

#     plt.subplot(1, 2, 1)
#     plt.imshow(before_vis)
#     plt.title(f"BEFORE - {sample['source_size']}")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(after_vis)
#     plt.title("AFTER - 720x480")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()


# print("\n========================")
# print("Resize completed.")
# print("========================")