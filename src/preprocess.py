import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def resize_and_pad_coco(json_path, image_dir, output_dir, target_size=512):
    # ساخت پوشه‌های خروجی
    output_image_dir = os.path.join(output_dir, "images")
    os.makedirs(output_image_dir, exist_ok=True)
    
    # لود کردن فایل JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    new_images = []
    new_annotations = []

    print(f"Processing images from {image_dir}...")
    
    for img_info in tqdm(coco_data['images']):
        img_path = os.path.join(image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        
        # ۱. محاسبه مقیاس (Scale)
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # ۲. تغییر اندازه تصویر
        resized_img = cv2.resize(image, (new_w, new_h))
        
        # ۳. محاسبه پدینگ برای وسط‌چین کردن
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        
        # ایجاد تصویر سیاه نهایی و قرار دادن تصویر تغییر یافته در مرکز
        final_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        final_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_img
        
        # ذخیره تصویر جدید
        cv2.imwrite(os.path.join(output_image_dir, img_info['file_name']), final_img)

        # ۴. آپدیت اطلاعات تصویر در JSON
        new_img_info = img_info.copy()
        new_img_info['width'] = target_size
        new_img_info['height'] = target_size
        new_images.append(new_img_info)

        # ۵. آپدیت تمام آنوتیشن‌های مربوط به این تصویر
        img_id = img_info['id']
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id:
                new_ann = ann.copy()
                
                # آپدیت Segmentation (پلی‌گون‌ها)
                new_seg = []
                for poly in ann['segmentation']:
                    # تبدیل لیست [x1, y1, x2, y2, ...]
                    poly_np = np.array(poly).reshape(-1, 2)
                    poly_np = poly_np * scale # اعمال مقیاس
                    poly_np[:, 0] += pad_w    # اعمال پدینگ افقی
                    poly_np[:, 1] += pad_h    # اعمال پدینگ عمودی
                    new_seg.append(poly_np.reshape(-1).tolist())
                new_ann['segmentation'] = new_seg
                
                # آپدیت Bounding Box [x, y, width, height]
                bbox = np.array(ann['bbox'])
                new_bbox = [
                    bbox[0] * scale + pad_w, # x_new
                    bbox[1] * scale + pad_h, # y_new
                    bbox[2] * scale,         # w_new
                    bbox[3] * scale          # h_new
                ]
                new_ann['bbox'] = new_bbox
                new_ann['area'] = new_bbox[2] * new_bbox[3] # آپدیت مساحت
                new_annotations.append(new_ann)

    # ذخیره فایل JSON نهایی
    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_data['categories']
    }
    
    with open(os.path.join(output_dir, "annotations.json"), 'w') as f:
        json.dump(new_coco, f)

    print(f"\nDone! Preprocessed data saved to {output_dir}")

# مثال برای اجرا (باید مسیرهای خودت را جایگزین کنی)
resize_and_pad_coco("C:/Users/LENOVO/Desktop/Segmentation paper DG/data/Cataract-1K/_annotations.coco.json", "C:/Users/LENOVO/Desktop/Segmentation paper DG/data/Cataract-1K/test", "C:/Users/LENOVO/Desktop/Segmentation paper DG/data/Cataract-1K/After padding", 512)

