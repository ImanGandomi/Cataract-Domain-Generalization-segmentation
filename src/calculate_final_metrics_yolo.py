import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

# ==========================================
# بخش اول: محاسبه mAP از طریق خود YOLO
# ==========================================
print("--- Calculating mAP Metrics ---")
model = YOLO(r'C:\Users\LENOVO\Desktop\Segmentation paper DG\best.pt') # مسیر وزن‌های مدل خود را قرار دهید

# ارزیابی روی دیتای تست (مسیر فایل yaml دیتاسِت خود را بدهید)
metrics = model.val(data=r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\new\txt\padded_splited_data_114\data.yaml', split='test')

map50 = metrics.seg.map50
map50_95 = metrics.seg.map

print(f"mAP@0.5:      {map50:.4f}")
print(f"mAP@0.5:0.95: {map50_95:.4f}")
print("-" * 30)

# ==========================================
# بخش دوم: محاسبه پیکسل‌محور IoU و DSC
# ==========================================
def parse_yolo_txt_to_mask(txt_path, img_height, img_width):
    """تبدیل نقاط پلی‌گان در فایل txt یولو به ماسک باینری"""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    if not os.path.exists(txt_path):
        return mask
    
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            # خواندن نقاط و تبدیل از حالت نرمالایز شده به مختصات پیکسلی
            points = np.array(parts[1:], dtype=float).reshape(-1, 2)
            points[:, 0] *= img_width   # ضرب در عرض
            points[:, 1] *= img_height  # ضرب در ارتفاع
            points = points.astype(np.int32)
            
            # رسم پلی‌گان روی ماسک
            cv2.fillPoly(mask, [points], 1)
            
    return mask

def calculate_iou_and_dice(pred_mask, gt_mask):
    """محاسبه IoU و Dice برای دو ماسک باینری"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    pred_area = pred_mask.sum()
    gt_area = gt_mask.sum()
    
    iou = intersection / union if union > 0 else 0.0
    dice = (2.0 * intersection) / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 0.0
    
    return iou, dice

print("--- Calculating Mask IoU and DSC ---")

# مسیر پوشه تصاویر و لیبل‌های تست را مشخص کنید
test_images_path = r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\new\txt\padded_splited_data_114\test\images\*.jpg' # پسوند تصاویر خود را بررسی کنید (.png یا .jpg)
test_labels_dir = r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\CaDIS\new\txt\padded_splited_data_114\test\images'

total_iou = 0.0
total_dice = 0.0
valid_images_count = 0

image_files = glob.glob(test_images_path)

for img_path in image_files:
    # 1. خواندن تصویر اصلی
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 2. پیدا کردن فایل txt متناظر
    base_name = os.path.basename(img_path)
    txt_name = os.path.splitext(base_name)[0] + '.txt'
    txt_path = os.path.join(test_labels_dir, txt_name)
    
    # 3. ایجاد ماسک Ground Truth
    gt_mask = parse_yolo_txt_to_mask(txt_path, h, w)
    
    # 4. پیش‌بینی با مدل
    results = model.predict(img_path, verbose=False)
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    
    if results[0].masks is not None:
        # دریافت ماسک پیش‌بینی شده (ریسایز شده به ابعاد تصویر اصلی)
        masks = results[0].masks.data.cpu().numpy()
        
        # ادغام تمام ماسک‌های پیش‌بینی شده در یک تصویر در صورت وجود چند آبجکت
        for m in masks:
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_mask = np.logical_or(pred_mask, m_resized).astype(np.uint8)
    
    # 5. محاسبه معیارها
    iou, dice = calculate_iou_and_dice(pred_mask, gt_mask)
    
    total_iou += iou
    total_dice += dice
    valid_images_count += 1

# محاسبه میانگین کل
mean_iou = total_iou / valid_images_count if valid_images_count > 0 else 0
mean_dice = total_dice / valid_images_count if valid_images_count > 0 else 0

print(f"Total Test Images Analyzed: {valid_images_count}")
print(f"Mean Mask IoU: {mean_iou:.4f}")
print(f"Mean Mask DSC: {mean_dice:.4f}")