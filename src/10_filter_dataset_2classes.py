import shutil
from pathlib import Path

def find_exact_two_classes(labels_dir, class_a, class_b):
    target_classes = {str(class_a), str(class_b)}
    matching_files = []

    labels_path = Path(labels_dir)

    for txt_file in labels_path.glob("*.txt"):
        if txt_file.name == "classes.txt":
            continue

        present_classes = set()

        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        present_classes.add(parts[0])

            if present_classes == target_classes:
                matching_files.append(txt_file.stem)

        except Exception as e:
            print(f"Error reading file {txt_file.name}: {e}")

    return matching_files


# ==========================
# تنظیمات
# ==========================



LABELS_DIRECTORY = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\padded_splited_data_114\train\labels"
IMAGES_DIRECTORY = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\padded_splited_data_114\train\images"

OUTPUT_IMAGES = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\backgrounds\LMM\images"
OUTPUT_LABELS = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\backgrounds\LMM\labels"

# ساخت پوشه‌های مقصد
Path(OUTPUT_IMAGES).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_LABELS).mkdir(parents=True, exist_ok=True)

CLASS_1 = 3
CLASS_2 = 9

# نتایج تابع قبلی
results = find_exact_two_classes(
    LABELS_DIRECTORY,
    CLASS_1,
    CLASS_2
)

# ساخت پوشه‌های مقصد
Path(OUTPUT_IMAGES).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_LABELS).mkdir(parents=True, exist_ok=True)

copied_count = 0

for filename in results:

    # ---------- کپی فایل txt ----------
    src_label = Path(LABELS_DIRECTORY) / f"{filename}.txt"

    if src_label.exists():
        dst_label = Path(OUTPUT_LABELS) / src_label.name
        shutil.copy2(src_label, dst_label)

    # ---------- کپی تصویر ----------
    image_found = False

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        src_image = Path(IMAGES_DIRECTORY) / f"{filename}{ext}"

        if src_image.exists():
            dst_image = Path(OUTPUT_IMAGES) / src_image.name
            shutil.copy2(src_image, dst_image)

            copied_count += 1
            image_found = True
            break

    if not image_found:
        print(f"Image not found for {filename}")

print(f"\nCopied {copied_count} image-label pairs.")







