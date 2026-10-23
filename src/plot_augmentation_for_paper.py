import cv2
import os
import albumentations as A

def generate_paper_augmentations(image_path, output_folder):
    # بررسی وجود تصویر اصلی
    if not os.path.exists(image_path):
        print(f"خطا: تصویر در مسیر '{image_path}' پیدا نشد!")
        return

    # ایجاد پوشه خروجی
    os.makedirs(output_folder, exist_ok=True)

    # خواندن تصویر و تبدیل به RGB
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # رنگ پس‌زمینه درخواستی (R, G, B)
    bg_color = (114, 114, 114)

    # تعریف مقادیر فیکس و مشخص برای دیده شدن واضح در مقاله
    augmentations = {
        "1_fliplr": A.HorizontalFlip(p=1.0),
        
        # چرخش دقیقاً روی ۲۰ درجه (حذف حالت تصادفی با تنظیم حد بالا و پایین روی 20)
        "2_degrees": A.Rotate(
            limit=(15, 15), 
            border_mode=cv2.BORDER_CONSTANT, 
            value=bg_color, 
            fill=(114, 114, 114), # 🌟 رفع مشکل مشکی بودن: برای نسخه‌های جدید
            cval=(114, 114, 114), # 🌟 برای سازگاری با نسخه‌های قدیمی‌تر
            p=1.0
        ),
        
        # جابجایی دقیقاً ۱۵ درصد در محور X و Y
        "3_translate": A.Affine(
            translate_percent={"x": (0.15, 0.15), "y": (0.15, 0.15)}, 
            cval=bg_color, 
            fill=(114, 114, 114), # 🌟 رفع مشکل مشکی بودن: برای نسخه‌های جدید
            p=1.0
        ),
        
        # تغییر اندازه ثابت و همسان برای هر دو بعد (کوچک شدن تا ۶۵ درصد اندازه اصلی)
        "4_scale": A.Affine(
            scale=(1.15, 1.15), 
            cval=bg_color, 
            fill=(114, 114, 114), # 🌟 رفع مشکل مشکی بودن: برای نسخه‌های جدید
            p=1.0
        ),
        
        # افزایش اشباع رنگ به یک میزان ثابت (تا رنگ‌ها کمی غلیظ‌تر دیده شوند)
        "5_hsv_s": A.HueSaturationValue(
            hue_shift_limit=0, 
            sat_shift_limit=(30, 30), 
            val_shift_limit=0, 
            fill=(114, 114, 114), # 🌟 رفع مشکل مشکی بودن: برای نسخه‌های جدید
            p=1.0
        ),
        
        # کاهش روشنایی به میزان ثابت (برای نمایش تاریک‌تر شدن تصویر)
        "6_hsv_v": A.HueSaturationValue(
            hue_shift_limit=0, 
            sat_shift_limit=0, 
            val_shift_limit=(-30, -30), 
            fill=(114, 114, 114), # 🌟 رفع مشکل مشکی بودن: برای نسخه‌های جدید
            cval=(114, 114, 114), # 🌟 برای سازگاری با نسخه‌های قدیمی‌تر
            p=1.0
        ),
    }

    # اعمال، تبدیل مجدد به BGR و ذخیره
    for name, aug_func in augmentations.items():
        augmented = aug_func(image=image_rgb)
        result_rgb = augmented["image"]
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(output_folder, f"{name}.jpg")
        cv2.imwrite(save_path, result_bgr)
        print(f"تصویر '{name}' آماده و ذخیره شد.")

# ==========================================
# نحوه اجرا
# ==========================================
input_img = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\plot_augmentation/Picture3.jpg"
# نام پوشه‌ای که می‌خواهید عکس‌ها آنجا ذخیره شوند
out_dir = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\plot_augmentation/aa"

generate_paper_augmentations(input_img, out_dir)
