import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

def get_distinct_colors(num_colors):
    """ایجاد رنگ‌های تصادفی ولی متمایز (BGR)"""
    colors = []
    for i in range(num_colors):
        # استفاده از فضای رنگی HSV برای ایجاد رنگ‌های متمایز و تبدیل به BGR
        hue = int(180 * i / num_colors)
        sat = 255
        val = 255
        hsv_color = np.uint8([[[hue, sat, val]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr_color))
    return colors

def verify_batch_labels_colored(image_dir, json_path, num_samples=10):
    # ۱. لود کردن فایل JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # ۲. ساخت نقشه‌برداری کلاس‌ها (Class ID -> Class Name)
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    cat_ids = sorted(categories.keys())
    
    # ۳. ایجاد رنگ‌های مجزا برای هر کلاس
    class_colors = {cat_ids[i]: color for i, color in enumerate(get_distinct_colors(len(cat_ids)))}
    
    images_info = coco_data['images']
    # اگر تعداد تصاویر کمتر از درخواست بود، همه را نشان بده
    num_samples = min(num_samples, len(images_info))
    
    # انتخاب ۱۰ تصویر تصادفی
    sample_images = random.sample(images_info, num_samples)
    
    # تنظیم ابعاد جدول نمایش (۲ سطر و ۵ ستون)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10)) # کمی عریض‌تر برای جا شدن متن‌ها
    axes = axes.flatten()

    for i, img_info in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # ما مستقیماً روی تصویر BGR رسم می‌کنیم و در آخر به RGB تبدیل می‌کنیم
        
        img_id = img_info['id']
        
        # پیدا کردن و رسم تمام آنوتیشن‌های این تصویر
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id:
                cat_id = ann['category_id']
                class_name = categories.get(cat_id, "Unknown")
                color = class_colors.get(cat_id, (255, 255, 255)) # پیش‌فرض سفید اگر کلاس ناشناخته بود
                
                # رسم Bounding Box (با رنگ مخصوص کلاس)
                bbox = [int(v) for v in ann['bbox']]
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
                
                # رسم Segmentation (آبی کم‌رنگ‌تر برای شفافیت، یا همان رنگ کلاس)
                # فعلا از همان رنگ کلاس استفاده می‌کنیم ولی با ضخامت ۱
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(img, [poly], True, color, 1)
                
                # ۴. اضافه کردن متن نام کلاس بالای کادر مرزی
                # تنظیم مختصات متن (اندکی بالاتر از کادر)
                text_org = (bbox[0], max(15, bbox[1] - 5))
                # تنظیم پس‌زمینه متن برای خوانایی بیشتر (رنگ کلاس معکوس یا سیاه)
                # برای سادگی، فعلا یک مستطیل کوچک سیاه می‌کشیم
                (text_w, text_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (text_org[0], text_org[1] - text_h - 2), (text_org[0] + text_w, text_org[1] + 2), color, -1)
                # متن سفید یا سیاه روی پس‌زمینه کلاس
                cv2.putText(img, class_name, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        # تبدیل به RGB برای نمایش در Matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Image ID: {img_id}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# مسیرها را مطابق با خروجی preprocess.py تنظیم کن
verify_batch_labels_colored(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\with_padding\images", r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\with_padding\annotations.json")