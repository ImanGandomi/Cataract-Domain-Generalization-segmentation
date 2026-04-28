import os
from collections import Counter

def check_yolo_labels(folder_path):
    """
    بررسی بازه اعداد کلاس‌ها در فایل‌های YOLO TXT
    """
    all_classes = []
    file_count = 0
    
    # بررسی وجود پوشه
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        return

    print(f"🔍 Scanning labels in: {folder_path} ...")

    # پیمایش تمام فایل‌های txt
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_count += 1
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        # اولین بخش هر سطر آیدی کلاس است
                        class_id = int(parts[0])
                        all_classes.append(class_id)

    if not all_classes:
        print("⚠️ No labels found in the files.")
        return

    unique_classes = sorted(list(set(all_classes)))
    counts = Counter(all_classes)

    print("\n" + "="*30)
    print("📊 LABEL ANALYSIS REPORT")
    print("="*30)
    print(f"📁 Total Files Scanned: {file_count}")
    print(f"🔢 Range Found: {min(unique_classes)} to {max(unique_classes)}")
    print(f"✨ Unique Class IDs: {unique_classes}")
    print(f"📦 Total Instances: {len(all_classes)}")
    print("-" * 30)
    print("📈 Distribution (Class: Count):")
    for cls in unique_classes:
        print(f"  Class {cls}: {counts[cls]} instances")
    print("="*30)

# --- مسیر پوشه لیبل‌های خود را اینجا جایگزین کنید ---
LABEL_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\txt\padded_splited_data\train\labels"

if __name__ == "__main__":
    check_yolo_labels(LABEL_DIR)