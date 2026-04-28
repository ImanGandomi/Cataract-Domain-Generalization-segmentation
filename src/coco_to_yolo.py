import json
import os
import shutil # برای پاکسازی پوشه
from tqdm import tqdm

def convert_to_yolo(json_path, images_root, output_labels_dir):
    # ۱. پاکسازی پوشه برای جلوگیری از تکرار آنوتیشن‌ها در دفعات اجرای مجدد
    if os.path.exists(output_labels_dir):
        shutil.rmtree(output_labels_dir)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    
    # ۲. مپینگ کلاس‌ها (نادیده گرفتن 0 و چیدمان 1 تا 12 به 0 تا 11)
    valid_cats = sorted([cat['id'] for cat in coco_data['categories'] if cat['id'] != 0])
    cat_mapping = {cat_id: i for i, cat_id in enumerate(valid_cats)}
    print(f"✅ Mapping applied: {cat_mapping}")
    
    for ann in tqdm(coco_data['annotations'], desc=f"Converting {os.path.basename(json_path)}"):
        if ann['category_id'] == 0:
            continue
        
        img_id = ann['image_id']
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        w, h = img_info['width'], img_info['height']
        
        label_name = os.path.splitext(img_info['file_name'])[0] + ".txt"
        label_path = os.path.join(output_labels_dir, label_name)
        
        class_id = cat_mapping[ann['category_id']]
        
        if 'segmentation' in ann:
            for seg in ann['segmentation']:
                if len(seg) >= 6:
                    # ۳. استفاده از فرمت .6f برای دقت بالاتر (استاندارد YOLO)
                    normalized_coords = []
                    for i in range(0, len(seg), 2):
                        nx = max(0.0, min(1.0, seg[i] / w))   # Clipping برای اطمینان
                        ny = max(0.0, min(1.0, seg[i+1] / h))
                        normalized_coords.append(f"{nx:.6f}")
                        normalized_coords.append(f"{ny:.6f}")
                    
                    line = f"{class_id} " + " ".join(normalized_coords) + "\n"
                    
                    with open(label_path, 'a') as f_out:
                        f_out.write(line)

# نحوه اجرا برای Cataract-1k (مثال):
base_path = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\coco\padded_splited_data\labels"
convert_to_yolo(f"{base_path}/train_annotations.json", f"{base_path}/images/train", r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\txt\padded_splited_data\train\labels")
convert_to_yolo(f"{base_path}/val_annotations.json", f"{base_path}/images/val", r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\txt\padded_splited_data\val\labels")
convert_to_yolo(f"{base_path}/test_annotations.json", f"{base_path}/images/test", r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\txt\padded_splited_data\test\labels")

