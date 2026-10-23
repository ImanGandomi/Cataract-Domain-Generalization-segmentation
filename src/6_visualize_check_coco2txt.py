import cv2
import os
import random
import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_dual_comparison(image_dir, label_dir, json_path, num_samples=10):
    # ۱. بارگذاری JSON و اعمال منطق فیلتر مشابه کد تبدیل
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # استخراج کلاس‌های معتبر (حذف ID 0) و مرتب‌سازی
    valid_categories = sorted([cat for cat in coco_data['categories'] if cat['id'] != 0], key=lambda x: x['id'])
    
    # ساخت لیست نام کلاس‌ها (ایندکس 0 این لیست حالا برابر با original ID 1 است)
    class_names = [cat['name'] for cat in valid_categories]
    
    # نگاشت برای رسم دیتای COCO: {Original_ID: YOLO_Index}
    # برای اینکه رنگ‌های COCO و YOLO در تصویر یکی شوند
    coco_id_to_yolo_idx = {cat['id']: i for i, cat in enumerate(valid_categories)}
    
    print(f"✅ Mapping for Visualization: { {cat['id']: i for i, cat in enumerate(valid_categories)} }")

    # نگاشت تصویر به آنوتیشن‌های COCO
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == 0: continue # نادیده گرفتن کلاس 0
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
        
    coco_images = coco_data['images']
    sample_images = random.sample(coco_images, min(num_samples, len(coco_images)))

    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    # ایجاد رنگ ثابت برای ۱۲ کلاس
    random.seed(42)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(12)]

    for i, img_info in enumerate(sample_images):
        img_name = img_info['file_name']
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        current_img_classes = []

        # --- رسم YOLO (Thick Lines) ---
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    yolo_idx = int(parts[0])
                    # حالا yolo_idx 0 به درستی به Cannula اشاره می‌کند
                    current_img_classes.append(class_names[yolo_idx])
                    
                    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    poly = coords.astype(np.int32)
                    cv2.polylines(img, [poly], isClosed=True, color=colors[yolo_idx], thickness=4)

        # --- رسم COCO اصلی (Thin White Lines) ---
        img_id = img_info['id']
        if img_id in img_id_to_anns:
            for ann in img_id_to_anns[img_id]:
                if 'segmentation' in ann:
                    for seg in ann['segmentation']:
                        poly_coco = np.array(seg, dtype=np.int32).reshape(-1, 2)
                        cv2.polylines(img, [poly_coco], isClosed=True, color=(255, 255, 255), thickness=1)

        unique_classes = sorted(list(set(current_img_classes)))
        title_text = ", ".join(unique_classes) if unique_classes else "No Instruments"
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(title_text, fontsize=9)
        axes[i].axis('off')

    plt.suptitle("Comparison: Thick Color = YOLO | Thin White = COCO JSON", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

# اجرا با مسیرهای ارسالی شما:
visualize_dual_comparison(
    image_dir=r"C:\Users\LENOVO\Desktop\Master thesis\Dataset\coco\splitted_data\images\test",
    label_dir=r"C:\Users\LENOVO\Desktop\Master thesis\Dataset\txt\splited_data\test\labels",
    json_path=r"C:\Users\LENOVO\Desktop\Master thesis\Dataset\coco\splitted_data\labels\test_annotations.json"
)