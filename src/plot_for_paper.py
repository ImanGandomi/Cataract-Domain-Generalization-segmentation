################################################################ for plot mask
import cv2
import numpy as np
import os
import glob

# رنگ‌های مشخص شده شما (در فرمت BGR برای OpenCV)
CLASS_ID_COLORS = {
    0: (251, 41, 1),      # Cannula
    1: (235, 218, 13),    # Cap-Cystotome
    2: (243, 242, 242),   # Cap-Forceps
    3: (182, 221, 4),     # Cornea
    4: (106, 31, 17),     # Forceps
    5: (221, 111, 255),   # IA-Handpiece
    6: (78, 65, 251),     # Lens-Injector
    7: (1, 235, 205),     # Phaco-Handpiece
    8: (68, 242, 0),      # Primary-Knife
    9: (255, 0, 190),     # Pupil
    10: (255, 180, 0),    # Second-Instrument
    11: (186, 1, 217)     # Secondary-Knife
}

def generate_masks(input_dir, output_dir):
    # ساخت پوشه خروجی در صورت عدم وجود
    os.makedirs(output_dir, exist_ok=True)
    
    # پیدا کردن تمام تصاویر در پوشه ورودی (پشتیبانی از فرمت‌های رایج)
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        
        # خواندن تصویر فقط برای استخراج ابعاد آن (طول و عرض)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_name}")
            continue
        height, width = img.shape[:2]
        
        # ساخت یک بوم کاملاً سیاه به اندازه تصویر اصلی
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # اگر فایل لیبل وجود نداشت، همان ماسک سیاه ذخیره می‌شود
        if not os.path.exists(txt_path):
            cv2.imwrite(os.path.join(output_dir, img_name), mask)
            continue
            
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        objects = []
        for line in lines:
            parts = line.strip().split()
            if not parts: 
                continue
                
            class_id = int(parts[0])
            
            # استخراج مختصات (نرمالایز شده در فرمت YOLO)
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            
            # تبدیل مختصات نرمالایز شده [0, 1] به مختصات پیکسلی واقعی
            coords[:, 0] *= width
            coords[:, 1] *= height
            coords = coords.astype(np.int32)
            
            objects.append({'class_id': class_id, 'polygon': coords})
            
        # ---------------------------------------------------------
        # مدیریت ترتیب رسم: اول Cornea (3) بعد Pupil (9) بعد بقیه
        # ---------------------------------------------------------
        def get_draw_priority(obj):
            cid = obj['class_id']
            if cid == 3: return 0    # بالاترین اولویت رسم (لایه زیرین)
            if cid == 9: return 1    # اولویت دوم
            return 2                 # سایر ابزارها (روی بقیه رسم می‌شوند)
            
        objects.sort(key=get_draw_priority)
        
        # رسم پلی‌گان‌ها (چندضلعی‌ها) روی بوم سیاه
        for obj in objects:
            cid = obj['class_id']
            if cid in CLASS_ID_COLORS:
                color = CLASS_ID_COLORS[cid]
                poly = obj['polygon']
                # پر کردن ماسک با رنگ مشخص شده
                cv2.fillPoly(mask, [poly], color)
                
        # ذخیره خروجی
        output_path = os.path.join(output_dir, f"mask_{img_name}")
        cv2.imwrite(output_path, mask)
        print(f"Processed and saved: {output_path}")

# ==========================================
# نحوه استفاده از کد:
# ==========================================
# مسیر پوشه‌ای که عکس‌ها و فایل‌های txt در آن قرار دارند
INPUT_FOLDER = r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\mask_plot' 

# مسیری که می‌خواهید ماسک‌های تولید شده در آن ذخیره شوند
OUTPUT_FOLDER = r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\mask_plot/ss3'  

# اجرای تابع
generate_masks(INPUT_FOLDER, OUTPUT_FOLDER)



#################################################################################### for extract just instrument
# import cv2
# import numpy as np
# import os
# import glob

# def process_dataset_to_masks_and_cutouts(input_dir, output_dir):
#     # ساخت پوشه‌های خروجی مجزا
#     white_mask_dir = os.path.join(output_dir, 'white_masks')
#     transparent_cutout_dir = os.path.join(output_dir, 'transparent_cutouts') # تغییر نام پوشه به شفاف
    
#     os.makedirs(white_mask_dir, exist_ok=True)
#     os.makedirs(transparent_cutout_dir, exist_ok=True)
    
#     # پیدا کردن تمام تصاویر در پوشه ورودی
#     image_paths = []
#     for ext in ('*.jpg', '*.jpeg', '*.png'):
#         image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
#     for img_path in image_paths:
#         img_name = os.path.basename(img_path)
#         img_name_wo_ext = os.path.splitext(img_name)[0] # نام فایل بدون پسوند
#         txt_path = os.path.splitext(img_path)[0] + '.txt'
        
#         # خواندن تصویر اصلی
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Error reading image: {img_name}")
#             continue
#         height, width = img.shape[:2]
        
#         # ۱. ساخت بوم سیاه برای ماسک سفید (۳ کاناله)
#         white_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
#         # ۲. ساخت بوم کاملاً شفاف (۴ کاناله - BGRA)
#         # همه پیکسل‌ها در ابتدا (0, 0, 0, 0) هستند که یعنی کاملاً شفاف
#         transparent_cutout = np.zeros((height, width, 4), dtype=np.uint8)
        
#         if not os.path.exists(txt_path):
#             cv2.imwrite(os.path.join(white_mask_dir, f"white_mask_{img_name}"), white_mask)
#             cv2.imwrite(os.path.join(transparent_cutout_dir, f"cutout_{img_name_wo_ext}.png"), transparent_cutout)
#             continue
            
#         with open(txt_path, 'r') as f:
#             lines = f.readlines()
            
#         for line in lines:
#             parts = line.strip().split()
#             if not parts: 
#                 continue
                
#             class_id = int(parts[0])
            
#             # نادیده گرفتن کلاس‌های Cornea (3) و Pupil (9)
#             if class_id in [3, 9]:
#                 continue
                
#             coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            
#             # تبدیل مختصات نرمالایز شده به پیکسلی
#             coords[:, 0] *= width
#             coords[:, 1] *= height
#             coords = coords.astype(np.int32)
            
#             poly = coords
            
#             # الف) رسم روی ماسک سفید
#             cv2.fillPoly(white_mask, [poly], (255, 255, 255))
            
#             # ب) انتقال تصویر واقعی ابزار روی بوم شفاف
#             temp_mask = np.zeros((height, width), dtype=np.uint8)
#             cv2.fillPoly(temp_mask, [poly], 255)
            
#             # کپی کردن رنگ‌ها (RGB) به ۳ کانال اول بوم شفاف
#             transparent_cutout[temp_mask == 255, 0:3] = img[temp_mask == 255]
#             # تنظیم کانال چهارم (Alpha) به 255 برای اینکه این بخش غیرشفاف (نمایان) شود
#             transparent_cutout[temp_mask == 255, 3] = 255
                
#         # ذخیره خروجی‌ها (خروجی کات‌اوت حتماً به صورت png ذخیره می‌شود)
#         cv2.imwrite(os.path.join(white_mask_dir, f"white_mask_{img_name}"), white_mask)
#         cv2.imwrite(os.path.join(transparent_cutout_dir, f"cutout_{img_name_wo_ext}.png"), transparent_cutout)
#         print(f"Successfully processed: {img_name}")

# # ==========================================
# # نحوه استفاده:
# # ==========================================
# INPUT_FOLDER = r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\extract_just_instrument'
# OUTPUT_FOLDER = r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\extract_just_instrument/B'

# process_dataset_to_masks_and_cutouts(INPUT_FOLDER, OUTPUT_FOLDER)




######################################################################################## plot background aug
# import cv2
# import numpy as np
# import os
# import glob
# import random

# class ImageAugmentor:
#     def __init__(self):
#         pass

#     def get_valid_y_range(self, img):
#         """ پیدا کردن محدوده بافت واقعی (بدون پدینگ‌های خاکستری 114 بالا و پایین) """
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         diff = np.abs(gray.astype(np.float32) - 114.0)
#         valid_rows = np.where(np.mean(diff, axis=1) > 5.0)[0]
        
#         if len(valid_rows) > 0:
#             y_min = valid_rows[0]
#             y_max = valid_rows[-1]
#             return y_min, y_max
            
#         return 0, img.shape[0]

#     def apply_domain_style(self, bg_img, valid_y_min, valid_y_max):
#         """ اعمال تغییرات نوری فقط روی بافت چشم (بدون دست زدن به پدینگ) """
#         img_array = bg_img.copy().astype(np.float32)
#         active_region = img_array[valid_y_min:valid_y_max, :, :]
        
#         if random.random() > 0.3:
#             color_shift = np.random.uniform(-15, 15, 3)
#             active_region[:, :, 0] += color_shift[0]
#             active_region[:, :, 1] += color_shift[1]
#             active_region[:, :, 2] += color_shift[2]
            
#         if random.random() > 0.3:
#             alpha = random.uniform(0.8, 1.2)
#             beta = random.uniform(-20, 20)
#             active_region = active_region * alpha + beta
            
#         if random.random() > 0.5:
#             blur_kernel = random.choice([3, 5])
#             active_region = cv2.GaussianBlur(active_region, (blur_kernel, blur_kernel), 0)

#         img_array[valid_y_min:valid_y_max, :, :] = active_region
#         return np.clip(img_array, 0, 255).astype(np.uint8)

#     def process_folder(self, input_dir, output_dir, num_aug_per_img=8):
#         """ خواندن تصاویر و ایجاد ۸ نسخه تغییر یافته برای هر کدام """
#         os.makedirs(output_dir, exist_ok=True)
        
#         image_paths = []
#         for ext in ('*.jpg', '*.jpeg', '*.png'):
#             image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
            
#         if not image_paths:
#             print("هیچ تصویری در پوشه ورودی پیدا نشد!")
#             return

#         print(f"تعداد {len(image_paths)} تصویر پیدا شد. شروع فرآیند Augmentation (ایجاد {num_aug_per_img} نسخه برای هر عکس)...")

#         for img_path in image_paths:
#             img_name = os.path.basename(img_path)
            
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"خطا در خواندن تصویر: {img_name}")
#                 continue
            
#             y_min, y_max = self.get_valid_y_range(img)
            
#             # حلقه برای ایجاد ۸ تصویر مجزا و تصادفی از روی عکس فعلی
#             for i in range(1, num_aug_per_img + 1):
#                 processed_img = self.apply_domain_style(img, y_min, y_max)
                
#                 # اضافه کردن شماره ایندکس به نام فایل خروجی (مثال: styled_v1_image.jpg)
#                 output_name = f"styled_v{i}_{img_name}"
#                 output_path = os.path.join(output_dir, output_name)
                
#                 cv2.imwrite(output_path, processed_img)
                
#             print(f"موفق: برای تصویر {img_name} تعداد {num_aug_per_img} نسخه ساخته شد.")


# # ==========================================
# # نحوه استفاده و اجرا:
# # ==========================================
# if __name__ == "__main__":
    
#     INPUT_FOLDER = r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\background_aug'
#     OUTPUT_FOLDER = r'C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\background_aug\a'
    
#     augmentor = ImageAugmentor()
#     # فراخوانی تابع (به طور پیش‌فرض روی عدد ۸ تنظیم شده است)
#     augmentor.process_folder(INPUT_FOLDER, OUTPUT_FOLDER, num_aug_per_img=8)
