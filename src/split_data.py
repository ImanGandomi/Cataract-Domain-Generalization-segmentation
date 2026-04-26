import json
import os
import random
import re
import shutil  # برای کپی کردن فایل‌ها
from collections import defaultdict
from tqdm import tqdm # برای نمایش نوار پیشرفت

def extract_video_id(filename, dataset_type):
    """
    استخراج ID ویدیو بر اساس الگوی هر دیتاست
    """
    if dataset_type == 'cataract-1k':
        # مطابق کد ارسالی شما: پیدا کردن عدد بعد از case
        match = re.search(r'case(\d+)', filename)
    elif dataset_type == 'lmm':
        match = re.search(r'SE_(\d+)', filename)
    elif dataset_type == 'cadis':
        match = re.search(r'Video_(\d+)', filename)
    else:
        return None
    return match.group(0) if match else None

def split_coco_by_video(json_path, input_images_dir, output_dir, dataset_type, train_ratio=0.7, val_ratio=0.15):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    video_to_images = defaultdict(list)
    skipped_count = 0

    for img in coco_data['images']:
        vid_id = extract_video_id(img['file_name'], dataset_type)
        if vid_id:
            video_to_images[vid_id].append(img)
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"⚠️ Warning: {skipped_count} images skipped (Pattern not found for {dataset_type})")

    all_videos = list(video_to_images.keys())
    random.seed(42)
    random.shuffle(all_videos)

    num_vids = len(all_videos)
    train_end = int(num_vids * train_ratio)
    val_end = train_end + int(num_vids * val_ratio)

    splits = {
        'train': all_videos[:train_end],
        'val': all_videos[train_end:val_end],
        'test': all_videos[val_end:]
    }

    print(f"\n--- Dataset: {dataset_type} ---")
    print(f"Total Unique Videos: {num_vids}")
    
    # ساخت ساختار درختی در خروجی
    os.makedirs(output_dir, exist_ok=True)

    for split_name, video_ids in splits.items():
        split_images = []
        target_img_ids = set()
        
        # ایجاد فولدر تصاویر برای هر ست (مثلاً output/images/train)
        split_img_output_dir = os.path.join(output_dir, "images", split_name)
        os.makedirs(split_img_output_dir, exist_ok=True)
        
        print(f"📂 Copying {split_name} images...")
        
        for vid in video_ids:
            for img in video_to_images[vid]:
                split_images.append(img)
                target_img_ids.add(img['id'])
                
                # عملیات کپی تصویر
                src_path = os.path.join(input_images_dir, img['file_name'])
                dst_path = os.path.join(split_img_output_dir, img['file_name'])
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path) # کپی همراه با حفظ متا‌دیتا
                else:
                    print(f"❌ File not found: {src_path}")
        
        # فیلتر کردن آنوتیشن‌ها
        split_annotations = [
            ann for ann in coco_data['annotations'] 
            if ann['image_id'] in target_img_ids
        ]
        
        new_coco = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": coco_data['categories']
        }
        
        # ذخیره فایل JSON در پوشه مخصوص (مثلاً output/labels/train_annotations.json)
        label_output_dir = os.path.join(output_dir, "labels")
        os.makedirs(label_output_dir, exist_ok=True)
        
        out_json_path = os.path.join(label_output_dir, f"{split_name}_annotations.json")
        with open(out_json_path, 'w') as f:
            json.dump(new_coco, f)
        
        print(f"✅ {split_name.capitalize()}: {len(video_ids)} vids, {len(split_images)} images copied.")

# --- تنظیمات اجرا ---
json_input = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\with_padding\annotations.json"
images_input = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\with_padding\images" # آدرس فولدر تصاویر پد شده
output_path = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\padded_splited_data"

split_coco_by_video(json_input, images_input, output_path, "cataract-1k")