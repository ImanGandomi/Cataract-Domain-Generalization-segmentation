from ultralytics import YOLO
import torch
import sys


def start_training():
    # ۱. انتخاب مدل YOLOv11-seg 
    # (n: nano, s: small, m: medium, l: large, x: extra large)
    # برای شروع و بنچمارک اولیه، مدل 's' تعادل خوبی بین سرعت و دقت دارد.
    model = YOLO("yolo11s-seg.pt") 

    # ۲. تنظیمات آموزش
    model.train(
        data=r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\txt\padded_splited_data\data.yaml",      # مسیر فایل تنظیمات
        epochs=100,            # تعداد دفعات تکرار (قابل تغییر بر اساس مانیتورینگ)
        imgsz=512,             # ابعاد تصاویر (مطابق با پدینگ انجام شده)
        batch=16,              # اگر با کمبود حافظه GPU مواجه شدید، این عدد را کم کنید
        device=0,              # استفاده از GPU (اگر چند GPU دارید، لیست بدهید [0, 1])
        project="Surgical_Seg", # نام پروژه برای ذخیره‌سازی
        name="cataract_v11_exp1", # نام این آزمایش خاص
        patience=20,           # Early Stopping (اگر ۲۰ اپوک پیشرفتی حاصل نشد، متوقف شو)
        save=True,             # ذخیره وزن‌های مدل
        plots=True,            # رسم نمودارهای Loss و mAP
        deterministic=True,    # برای تکرارپذیری نتایج در مقاله
        seed=42                # عدد شانس ما برای آزمایش‌های علمی
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