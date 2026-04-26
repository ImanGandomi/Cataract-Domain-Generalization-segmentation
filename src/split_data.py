import json
import os
import re
import shutil
from collections import defaultdict
from tqdm import tqdm

def extract_video_id(filename, dataset_type):
    """
    استخراج ID ویدیو و استانداردسازی آن برای Cataract-1k
    """
    if dataset_type == 'cataract-1k':
        # اصلاح شده برای پوشش هر دو حالت case و case_ مطابق اصلاحیه شما
        match = re.search(r'case_?(\d+)', filename)
        if match:
            # خروجی همیشه به صورت caseXXXX خواهد بود تا با لیست‌های شما مطابقت کند
            return f"case{match.group(1)}"
    elif dataset_type == 'lmm':
        match = re.search(r'SE_(\d+)', filename)
        if match: return match.group(0)
    elif dataset_type == 'cadis':
        match = re.search(r'Video_(\d+)', filename)
        if match: return match.group(0)
    return None

def split_coco_manual(json_path, input_images_dir, output_dir, dataset_type):
    # ۱. تعریف لیست‌های دستی شما
    # نکته: نام‌ها را بر اساس استاندارد خروجی تابع extract_video_id (بدون آندرلاین) می‌نویسیم
    test_vids = {'case5015', 'case5300', 'case5303', 'case5304', 'case5305', 'case5329'}
    val_vids = {'case5017', 'case5032', 'case5301', 'case5309', 'case53153', 'case5325'}

    with open(json_path, 'r', encoding='utf-8') as f:
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
        print(f"⚠️ Warning: {skipped_count} images skipped.")

    all_videos = list(video_to_images.keys())
    
    # ۲. تخصیص ویدیوها به دسته‌های مشخص شده
    splits = {'train': [], 'val': [], 'test': []}
    
    for vid in all_videos:
        if vid in test_vids:
            splits['test'].append(vid)
        elif vid in val_vids:
            splits['val'].append(vid)
        else:
            splits['train'].append(vid)

    print(f"\n--- Dataset: {dataset_type} (Manual Split) ---")
    print(f"Total Unique Videos Found: {len(all_videos)}")
    
    os.makedirs(output_dir, exist_ok=True)

    # ۳. فرآیند کپی و ساخت فایل‌های JSON
    for split_name, video_ids in splits.items():
        split_images = []
        target_img_ids = set()
        
        split_img_output_dir = os.path.join(output_dir, "images", split_name)
        os.makedirs(split_img_output_dir, exist_ok=True)
        
        print(f"📂 Processing {split_name} ({len(video_ids)} videos)...")
        
        for vid in video_ids:
            for img in video_to_images[vid]:
                split_images.append(img)
                target_img_ids.add(img['id'])
                
                src_path = os.path.join(input_images_dir, img['file_name'])
                dst_path = os.path.join(split_img_output_dir, img['file_name'])
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
        
        split_annotations = [
            ann for ann in coco_data['annotations'] 
            if ann['image_id'] in target_img_ids
        ]
        
        new_coco = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": coco_data['categories']
        }
        
        label_output_dir = os.path.join(output_dir, "labels")
        os.makedirs(label_output_dir, exist_ok=True)
        
        out_json_path = os.path.join(label_output_dir, f"{split_name}_annotations.json")
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_coco, f)
        
        print(f"✅ {split_name.capitalize()} Done: {len(split_images)} images.")

# --- تنظیمات اجرا ---
JSON_INPUT = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\with_padding\annotations.json"
IMAGES_INPUT = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\with_padding\images"
OUTPUT_PATH = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\padded_splited_data"

if __name__ == "__main__":
    split_coco_manual(JSON_INPUT, IMAGES_INPUT, OUTPUT_PATH, "cataract-1k")