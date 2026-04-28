import json
import os
from tqdm import tqdm

def convert_to_yolo(json_path, images_root, output_labels_dir):
    """
    تبدیل آنوتیشن‌های COCO به فرمت TXT برای YOLO Segmentation
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # ساخت نقشه‌برداری برای تصاویر
    images = {img['id']: img for img in coco_data['images']}
    
    # استخراج دسته‌بندی‌ها (YOLO باید از 0 شروع شود)
    # فرض بر این است که ID کلاس‌ها در COCO از 1 شروع شده، پس یکی کم می‌کنیم
    cat_mapping = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
    
    for ann in tqdm(coco_data['annotations'], desc=f"Converting {os.path.basename(json_path)}"):
        img_id = ann['image_id']
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        w, h = img_info['width'], img_info['height']
        
        # نام فایل لیبل باید با نام تصویر یکی باشد
        label_name = os.path.splitext(img_info['file_name'])[0] + ".txt"
        label_path = os.path.join(output_labels_dir, label_name)
        
        class_id = cat_mapping[ann['category_id']]
        
        # فرمت سگمنتیشن YOLO: class_id x1 y1 x2 y2 ... (نرمال شده بین 0 و 1)
        if 'segmentation' in ann:
            for seg in ann['segmentation']:
                if len(seg) >= 6: # حداقل 3 نقطه برای یک پلی‌گون
                    normalized_coords = []
                    for i in range(0, len(seg), 2):
                        normalized_coords.append(str(seg[i] / w))   # x
                        normalized_coords.append(str(seg[i+1] / h)) # y
                    
                    line = f"{class_id} " + " ".join(normalized_coords) + "\n"
                    
                    with open(label_path, 'a') as f_out:
                        f_out.write(line)

# نحوه اجرا برای Cataract-1k (مثال):
base_path = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\coco\padded_splited_data\labels"
convert_to_yolo(f"{base_path}/train_annotations.json", f"{base_path}/images/train", r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\txt\padded_splited_data\train\labels")
convert_to_yolo(f"{base_path}/val_annotations.json", f"{base_path}/images/val", r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\txt\padded_splited_data\val\labels")
convert_to_yolo(f"{base_path}/test_annotations.json", f"{base_path}/images/test", r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\txt\padded_splited_data\test\labels")

