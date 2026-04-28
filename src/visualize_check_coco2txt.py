import cv2
import os
import random
import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_dual_comparison(image_dir, label_dir, json_path, num_samples=10):
    # ۱. استخراج اطلاعات و نگاشت آنوتیشن‌های COCO
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    # ساخت دیکشنری برای دسترسی سریع: {category_id: name}
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # نگاشت تصویر به آنوتیشن‌هایش در COCO
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
        
    # لیست تصاویر موجود در JSON
    coco_images = coco_data['images']
    sample_images = random.sample(coco_images, min(num_samples, len(coco_images)))

    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    # ایجاد رنگ ثابت برای هر کلاس
    random.seed(42)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_names]

    for i, img_info in enumerate(sample_images):
        img_name = img_info['file_name']
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        current_img_classes = [] # برای اصلاح تیتر

        # --- بخش اول: رسم لیبل‌های YOLO (خطوط ضخیم - پیکسل‌های تبدیل شده) ---
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    c_id = int(parts[0])
                    current_img_classes.append(class_names[c_id])
                    
                    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    poly = coords.astype(np.int32)
                    
                    # رسم YOLO با ضخامت بیشتر (ضخامت ۴)
                    cv2.polylines(img, [poly], isClosed=True, color=colors[c_id], thickness=4)

        # --- بخش دوم: رسم لیبل‌های اصلی COCO (خطوط نازک سفید برای مقایسه) ---
        img_id = img_info['id']
        if img_id in img_id_to_anns:
            for ann in img_id_to_anns[img_id]:
                if 'segmentation' in ann:
                    for seg in ann['segmentation']:
                        poly_coco = np.array(seg, dtype=np.int32).reshape(-1, 2)
                        # رسم COCO با خط نازک سفید (ضخامت ۱) برای دیدن انطباق
                        cv2.polylines(img, [poly_coco], isClosed=True, color=(255, 255, 255), thickness=1)

        # اصلاح تیتر: حذف تکرار کلاس‌ها و نمایش در بالای هر عکس
        unique_classes = sorted(list(set(current_img_classes)))
        title_text = ", ".join(unique_classes) if unique_classes else "No Label"
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Classes: {title_text}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle("Comparison: Thick Color = YOLO | Thin White = COCO JSON", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()

# اجرا با مسیرهای ارسالی شما:
visualize_dual_comparison(
    image_dir=r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\txt\padded_splited_data\train\images",
    label_dir=r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\txt\padded_splited_data\train\labels",
    json_path=r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\padded_splited_data\labels\train_annotations.json"
)