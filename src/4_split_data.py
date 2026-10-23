import json
import os
import re
import shutil
from collections import defaultdict
from tqdm import tqdm

def extract_video_id(filename, dataset_type):
    """
    استخراج ID ویدیو بر اساس فرمت اختصاصی هر دیتاست
    """
    if dataset_type == 'cataract-1k':
        # برای هر دو حالت case_?(\d+)
        match = re.search(r'case_?(\d+)', filename)
        if match: return f"case{match.group(1)}"
        
    elif dataset_type == 'lmm':
        # برای فرمت SE_XXXX
        match = re.search(r'SE_(\d+)', filename)
        if match: return match.group(0)
        
    elif dataset_type == 'cadis':
        # برای فرمت Video5_frame002770_png_jpg
        match = re.search(r'Video(\d+)', filename)
        if match: return f"Video{match.group(1)}"
        
    return None

def get_manual_split_lists(dataset_type):
    """
    تعریف لیست‌های دستی دقیق برای هر دیتاست
    """
    if dataset_type == 'cataract-1k':
        # val = {'case5017', 'case5032', 'case5301', 'case5309', 'case53153', 'case5325'}
        # test = {'case5015', 'case5300', 'case5303', 'case5304', 'case5305', 'case5329'}

        # Cat1k_anatomy_instrument
        # fold_0
        # val = {'case5032', 'case5063', 'case5072', 'case5180', 'case5309', 'case5325'}
        # test = {'case5014', 'case5015', 'case5017', 'case5051', 'case5303', 'case5334'}

        # # fold_1
        # val = {'case5014', 'case5015', 'case5017', 'case5051', 'case5303', 'case5334'}
        # test = {'case5032', 'case5063', 'case5072', 'case5180', 'case5309', 'case5325'}

        # # fold_2
        # val = {'case5014', 'case5015', 'case5017', 'case5051', 'case5303', 'case5334'}
        # test = {'case5104', 'case5299', 'case5304', 'case5305', 'case5319', 'case5340'}

        # # fold_3
        # val = {'case5014', 'case5015', 'case5017', 'case5051', 'case5303', 'case5334'}
        # test = {'case5013', 'case5300', 'case5301', 'case53153', 'case5317', 'case5335'}

        # my custom split based on fold 0
        val = {'case5032', 'case5063', 'case5180', 'case5309', 'case5325'}
        test = {'case5014', 'case5015', 'case5017', 'case5051', 'case5303', 'case5334'}
        
    
    elif dataset_type == 'lmm':
        # اعداد بر اساس لیست ارسالی شما (فرمت SE_00XX)
        val_raw = [17, 44, 46, 56, 126, 137, 142, 11, 23, 27, 41, 102, 105, 110, 111, 123, 64, 76, 114, 116, 135, 141]   #81 went to train
        test_raw = [103, 5, 20, 22, 113, 124, 14, 15, 34, 45, 51, 62, 70, 78, 120, 71, 75, 80, 89, 94, 99, 139]     #106 went to train
        val = {f"SE_{i:04d}" for i in val_raw}
        test = {f"SE_{i:04d}" for i in test_raw}
        
    elif dataset_type == 'cadis':
        # هماهنگ با فرمت Video5 (بدون آندرلاین بین نام و عدد)
        val = {'Video5', 'Video7', 'Video16'}
        test = {'Video2', 'Video12', 'Video22'}
    else:
        val, test = set(), set()
    
    return val, test

def split_coco_unified(json_path, input_images_dir, output_dir, dataset_type):
    val_vids, test_vids = get_manual_split_lists(dataset_type)

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # گروه‌بندی
    video_to_images = defaultdict(list)
    for img in coco_data['images']:
        vid_id = extract_video_id(img['file_name'], dataset_type)
        if vid_id:
            video_to_images[vid_id].append(img)

    all_videos = list(video_to_images.keys())
    
    splits = {'train': [], 'val': [], 'test': []}
    for vid in all_videos:
        if vid in test_vids:
            splits['test'].append(vid)
        elif vid in val_vids:
            splits['val'].append(vid)
        else:
            splits['train'].append(vid)

    print(f"\n--- 📊 Dataset: {dataset_type} ---")
    print(f"Total Videos Found: {len(all_videos)}")
    print(f"Split Count -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    os.makedirs(output_dir, exist_ok=True)
    for split_name, video_ids in splits.items():
        split_images = []
        target_img_ids = set()
        
        img_out = os.path.join(output_dir, "images", split_name)
        os.makedirs(img_out, exist_ok=True)
        
        for vid in tqdm(video_ids, desc=f"Processing {split_name}"):
            for img in video_to_images[vid]:
                split_images.append(img)
                target_img_ids.add(img['id'])
                
                src = os.path.join(input_images_dir, img['file_name'])
                dst = os.path.join(img_out, img['file_name'])
                if os.path.exists(src):
                    shutil.copy2(src, dst)
        
        # ذخیره آنوتیشن‌ها
        split_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] in target_img_ids]
        new_coco = {"images": split_images, "annotations": split_anns, "categories": coco_data['categories']}
        
        label_dir = os.path.join(output_dir, "labels")
        os.makedirs(label_dir, exist_ok=True)
        with open(os.path.join(label_dir, f"{split_name}_annotations.json"), 'w') as f:
            json.dump(new_coco, f)

    print(f"✅ Finished splitting {dataset_type} successfully.")

# --- مثال اجرا برای CADIS ---
if __name__ == "__main__":
    split_coco_unified(
        json_path=r"C:\Users\LENOVO\Desktop\Master thesis\Dataset\coco\raw\annotations.json",
        input_images_dir=r"C:\Users\LENOVO\Desktop\Master thesis\Dataset\coco\raw\images",
        output_dir=r"C:\Users\LENOVO\Desktop\Master thesis\Dataset\coco\splitted_data",
        dataset_type="lmm"
    )