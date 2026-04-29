from ultralytics import YOLO
import torch
import sys


def start_training():
    # ۱. انتخاب مدل YOLOv11-seg 
    # (n: nano, s: small, m: medium, l: large, x: extra large)
    # برای شروع و بنچمارک اولیه، مدل 's' تعادل خوبی بین سرعت و دقت دارد.
    model = YOLO("yolo11l-seg.pt") 

    # ۲. تنظیمات آموزش
    model.train(
        data=r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\txt\padded_splited_data\data.yaml",      # مسیر فایل تنظیمات
        epochs=3,
        optimizer='auto',
        imgsz=512,
        batch=16,
        device=0,
        save=True,
        plots=True, 

        # --- Research Integrity ---
        project='Cataract_DG_Phase1',
        name='Train_CaDIS_Base', # این نام را در هر اجرا تغییر بده
        seed=42,
        deterministic=True,
        patience=15,

        # --- Class Imbalance ---
        cls_pw=0.5,
        overlap_mask=True,
    

        # --- فعال‌سازی آگمنتیشن پایه ---
        augment=True, 
        fliplr=0.5,     # احتمال ۵۰٪ برای فلیپ چپ و راست
        degrees=0.0,    # چرخش
        translate=0.0,  # جابجایی
        scale=0.0,      # تغییر اندازه تصادفی
        shear=0.0,      # قیچی کردن تصویر
        perspective=0.0,# تغییر پرسپکتیو
        flipud=0.0,     # فلیپ بالا و پایین
        hsv_h=0.0,      # تغییر رنگ (Hue)
        hsv_s=0.0,      # غلظت رنگ (Saturation)
        hsv_v=0.0,      # روشنایی (Value)
        mosaic=0.0,     # ترکیب ۴ تصویر (حتماً صفر شود)
        mixup=0.0,      # ترکیب ۲ تصویر روی هم
        copy_paste=0.0, # کپی ابزارها روی تصاویر دیگر
        auto_augment=None,
        erasing=0.0     # حذف تصادفی بخشی از تصویر
    )




def check_environment():
    print(f"--- Environment Check ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("❌ CRITICAL: GPU not found. Do not start training on CPU!")
        return False



if __name__ == "__main__":
    if check_environment():
        start_training()