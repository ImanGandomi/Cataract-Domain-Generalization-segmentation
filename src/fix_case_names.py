import json
import os
from tqdm import tqdm

def fix_cataract_filenames(json_path, image_dir):
    """
    تغییر نام فیزیکی تصاویر و اصلاح نام‌ها در فایل JSON (از case_ به case)
    """
    # ۱. بارگذاری فایل JSON
    if not os.path.exists(json_path):
        print(f"❌ خطا: فایل JSON در مسیر {json_path} پیدا نشد.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    print("🚀 در حال شروع فرآیند اصلاح نام‌ها...")
    
    updated_count = 0
    missing_files = 0

    # ۲. پیمایش روی لیست تصاویر در JSON
    for img_entry in tqdm(coco_data['images'], desc="Processing Images"):
        old_name = img_entry['file_name']
        
        # چک کردن وجود عبارت 'case_' در نام فایل
        if 'case_' in old_name:
            new_name = old_name.replace('case_', 'case')
            
            old_file_path = os.path.join(image_dir, old_name)
            new_file_path = os.path.join(image_dir, new_name)
            
            # الف: تغییر نام فیزیکی فایل روی هارد
            if os.path.exists(old_file_path):
                try:
                    os.rename(old_file_path, new_file_path)
                    
                    # ب: به‌روزرسانی نام در دیتای JSON (فقط اگر تغییر فایل موفق بود)
                    img_entry['file_name'] = new_name
                    updated_count += 1
                except Exception as e:
                    print(f"⚠️ خطای غیرمنتظره در تغییر نام {old_name}: {e}")
            else:
                # اگر فایل در پوشه نبود، فقط نام را در JSON اصلاح می‌کنیم (شاید قبلاً دستی عوض شده)
                img_entry['file_name'] = new_name
                missing_files += 1

    # ۳. ذخیره نهایی فایل JSON اصلاح شده
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)

    print("-" * 30)
    print(f"✅ عملیات با موفقیت پایان یافت.")
    print(f"📝 تعداد نام‌های اصلاح شده: {updated_count}")
    if missing_files > 0:
        print(f"❓ تعداد تصاویری که در پوشه نبودند اما در JSON اصلاح شدند: {missing_files}")
    print(f"💾 فایل JSON در مسیر قبلی به‌روزرسانی شد.")

# --- تنظیم مسیرهای دقیق شما ---
# مسیرها را مطابق با ساختار پروژه خودت نهایی کن
JSON_FILE = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\raw\_annotations.coco.json"
IMAGE_FOLDER = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\raw\train"

if __name__ == "__main__":
    fix_cataract_filenames(JSON_FILE, IMAGE_FOLDER)