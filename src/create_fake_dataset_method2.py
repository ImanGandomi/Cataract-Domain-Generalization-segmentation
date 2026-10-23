import cv2
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt


class AdvancedHybridCompositor:
    def __init__(self, cornea_id, tool_ids, out_dir):
        self.cornea_id = cornea_id
        self.tool_ids = tool_ids
        
        self.out_img_dir = os.path.join(out_dir, 'images')
        self.out_label_dir = os.path.join(out_dir, 'labels')
        os.makedirs(self.out_img_dir, exist_ok=True)
        os.makedirs(self.out_label_dir, exist_ok=True)

    # ---------------------------------------------------------
    # بخش اول: Domain Randomization (الهام گرفته از مقاله Rethinking)
    # ---------------------------------------------------------
    
    def apply_domain_style(self, bg_img, valid_y_min, valid_y_max):
        """ اعمال تغییرات نوری فقط روی بافت چشم (بدون دست زدن به پدینگ) """
        img_array = bg_img.copy().astype(np.float32)
        
        # استخراج فقط ناحیه بافت چشم
        active_region = img_array[valid_y_min:valid_y_max, :, :]
        
        if random.random() > 0.3:
            color_shift = np.random.uniform(-15, 15, 3)
            active_region[:, :, 0] += color_shift[0]
            active_region[:, :, 1] += color_shift[1]
            active_region[:, :, 2] += color_shift[2]
            
        if random.random() > 0.3:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            active_region = active_region * alpha + beta
            
        if random.random() > 0.5:
            blur_kernel = random.choice([3, 5])
            active_region = cv2.GaussianBlur(active_region, (blur_kernel, blur_kernel), 0)

        # بازگرداندن ناحیه تغییر یافته به تصویر اصلی (پدینگ ها ثابت می مانند)
        img_array[valid_y_min:valid_y_max, :, :] = active_region
        
        return np.clip(img_array, 0, 255).astype(np.uint8)

    # ... (توابع read_yolo_label و save_yolo_label و get_valid_y_range مانند قبل حفظ می‌شوند) ...
    
    def read_yolo_label(self, txt_path, img_w, img_h):
        """ خواندن فایل txt یولو و تبدیل مختصات نرمال به پیکسلی (Absolute) """
        labels = []
        if not os.path.exists(txt_path):
            return labels
            
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue # حداقل باید 3 نقطه (6 عدد) + 1 آیدی کلاس باشد
                
                class_id = int(parts[0])
                # تبدیل مختصات نرمال شده به مختصات پیکسلی واقعی
                poly_norm = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                poly_abs = poly_norm * np.array([img_w, img_h])
                labels.append({"class_id": class_id, "polygon": np.int32(poly_abs)})
        return labels

    def save_yolo_label(self, txt_path, labels, img_w, img_h):
        """ ذخیره لیبل‌های جدید در قالب فایل txt یولو (نرمال‌سازی مجدد) """
        with open(txt_path, 'w') as f:
            for label in labels:
                class_id = label['class_id']
                poly_abs = label['polygon']
                
                # نرمال‌سازی بین 0 و 1
                poly_norm = poly_abs.astype(np.float32) / np.array([img_w, img_h])
                
                # اطمینان از اینکه نقاط خارج از کادر نرفته باشند
                poly_norm = np.clip(poly_norm, 0.0, 1.0)
                
                # تبدیل به رشته با فرمت یولو
                poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly_norm])
                f.write(f"{class_id} {poly_str}\n")

    
    def get_valid_y_range(self, img):
        """ پیدا کردن محدوده بافت واقعی (بدون پدینگ‌های خاکستری 114 بالا و پایین) """
        # تبدیل تصویر به خاکستری (رنگ 114 در BGR معادل همان 114 در Grayscale است)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # محاسبه قدر مطلق اختلافِ هر پیکسل از عدد 114
        # (استفاده از float32 برای جلوگیری از خطای Overflow در محاسبات تفریق ضروری است)
        diff = np.abs(gray.astype(np.float32) - 114.0)
        
        # محاسبه میانگین اختلاف رنگ در هر سطر
        # اگر سطری تماماً پدینگ خاکستری باشد، میانگین اختلاف آن نزدیک به صفر است.
        # عدد 5.0 به عنوان حاشیه امن (Tolerance) برای نویزهای احتمالی فشرده‌سازی JPEG در نظر گرفته شده است.
        valid_rows = np.where(np.mean(diff, axis=1) > 5.0)[0]
        
        if len(valid_rows) > 0:
            y_min = valid_rows[0]
            y_max = valid_rows[-1]
            return y_min, y_max
            
        return 0, img.shape[0] # اگر کل تصویر خاکستری بود (محال است)، کل ارتفاع را برگردان


    def get_cornea_properties(self, labels, img_w, img_h):
        """ پیدا کردن مرکز و مساحت قرنیه (حفظ منطق بالینی شما) """
        cornea_polys = [lb['polygon'] for lb in labels if lb['class_id'] == self.cornea_id]
        if not cornea_polys:
            return None, None
        
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, cornea_polys, 1)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), M["m00"]
        return None, None

    def transform_tools(self, src_img, src_labels, bg_labels, bg_img):
        h, w = src_img.shape[:2]
        src_center, src_area = self.get_cornea_properties(src_labels, w, h)
        bg_center, bg_area = self.get_cornea_properties(bg_labels, w, h)

        if not src_center or not bg_center:
            scale, tx, ty = 1.0, 0, 0
        else:
            scale = np.sqrt(bg_area / src_area)
            cx_src, cy_src = src_center
            cx_bg, cy_bg = bg_center
            tx = cx_bg - scale * cx_src
            ty = cy_bg - scale * cy_src

        tool_labels = [lb for lb in src_labels if lb['class_id'] in self.tool_ids]
        if not tool_labels:
            return None, None, None

        valid_y_min, valid_y_max = self.get_valid_y_range(bg_img)

        # ==========================================
        # 1. محاسبه مرزهای کُل گروه ابزارها (Global Bounding Box)
        # ==========================================
        global_min_y = float('inf')
        global_max_y = float('-inf')
        temp_polys = []

        for lb in tool_labels:
            poly = lb['polygon'].astype(np.float32)
            new_poly = np.zeros_like(poly)
            new_poly[:, 0] = poly[:, 0] * scale + tx
            new_poly[:, 1] = poly[:, 1] * scale + ty
            temp_polys.append(new_poly)
            
            # پیدا کردن بالاترین و پایین‌ترین نقطه بین "تمام" ابزارها
            global_min_y = min(global_min_y, np.min(new_poly[:, 1]))
            global_max_y = max(global_max_y, np.max(new_poly[:, 1]))

        # ==========================================
        # 2. بررسی برخورد گروهی و محاسبه یک شیفتِ واحد
        # ==========================================
        group_height = global_max_y - global_min_y
        valid_height = valid_y_max - valid_y_min

        # اگر قدِ کل ابزارها از قدِ کل چشم بزرگتر باشد، کلاً این عکس را رد می‌کنیم
        if group_height > valid_height:
            return None, None, None

        shift_y = 0
        if global_min_y < valid_y_min:
            # کل گروه ابزارها از بالا زده بیرون -> هل دادن به پایین (مثبت)
            shift_y = valid_y_min - global_min_y 
        elif global_max_y > valid_y_max:
            # کل گروه ابزارها از پایین زده بیرون -> هل دادن به بالا (منفی)
            shift_y = valid_y_max - global_max_y 

        # اعمال شیفت نهایی به جابجایی کُل (ty)
        ty += shift_y

        # ==========================================
        # 3. ثبت نهایی پلی‌گان‌ها با شیفتِ گروهی
        # ==========================================
        warped_tool_labels = []
        for i, lb in enumerate(tool_labels):
            final_poly = temp_polys[i]
            final_poly[:, 1] += shift_y  # اعمال شیفت یکسان به همه
            warped_tool_labels.append({"class_id": lb['class_id'], "polygon": final_poly.astype(np.int32)})

        # آپدیت ماتریس انتقال با ty جدید
        M = np.float32([[scale, 0, tx], [0, scale, ty]])

        tool_mask = np.zeros((h, w), dtype=np.uint8)
        tool_polys = [lb['polygon'] for lb in tool_labels]
        cv2.fillPoly(tool_mask, tool_polys, 255)

        tool_img = cv2.bitwise_and(src_img, src_img, mask=tool_mask)
        warped_tool_img = cv2.warpAffine(tool_img, M, (w, h), flags=cv2.INTER_LINEAR)
        warped_tool_mask = cv2.warpAffine(tool_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        return warped_tool_img, warped_tool_mask, warped_tool_labels

    # ---------------------------------------------------------
    # بخش دوم: Gaussian Soft Blending (الهام گرفته از مقاله 1_Generalizing)
    # ---------------------------------------------------------
    # ==========================================
    # افزوده جدید: ترکیب پیشرفته و طبیعی (Gaussian Blending)
    # ==========================================
    def blend_gaussian(self, bg_img, bg_labels, warped_tool_img, warped_tool_mask, warped_tool_labels):
        valid_y_min, valid_y_max = self.get_valid_y_range(bg_img)
        
        # 1. اعمال استایل فقط روی بافت چشم (تضمین حفظ پدینگ 114)
        bg_styled = self.apply_domain_style(bg_img, valid_y_min, valid_y_max)

        # 2. سایش و بلور کردن ماسک (از بین بردن هاله‌های مصنوعی و تیز)
        kernel_erode = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(warped_tool_mask, kernel_erode, iterations=1)
        soft_mask = cv2.GaussianBlur(eroded_mask, (5, 5), 0)
        
        # 3. نرمال‌سازی ماسک برای ترکیب
        soft_mask = soft_mask.astype(np.float32) / 255.0
        soft_mask = np.expand_dims(soft_mask, axis=-1)

        bg_float = bg_styled.astype(np.float32)
        fg_float = warped_tool_img.astype(np.float32)

        # 4. آلفا بلندیگ خطی
        final_img = (bg_float * (1.0 - soft_mask)) + (fg_float * soft_mask)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        final_labels = bg_labels.copy()
        final_labels.extend(warped_tool_labels)

        return final_img, final_labels

# =================================================================
# نحوه فراخوانی در حلقه تولید دیتاست (دقیقاً مشابه اسکریپت قبلی شما):
# tool_data = compositor.transform_tools(src_img, src_labels, bg_labels, bg_img)
# w_tool_img, w_tool_mask, w_tool_labels = tool_data
# final_img, final_labels = compositor.blend_gaussian(bg_img, bg_labels, w_tool_img, w_tool_mask, w_tool_labels)
# =================================================================


# ==========================================
# بخش تولید دیتاست یولو
# ==========================================
def generate_yolo_dataset(SOURCES_DIR, BACKGROUNDS_DIR, OUTPUT_DIR):
    # ==========================================
    # ۱. آدرس‌دهی مسیرهای اصلی (این ۳ خط را تغییر دهید)
    # ==========================================
    
    

    # ==========================================
    # ۲. تنظیمات کلاس‌ها (بر اساس فایل classes.yaml شما)
    # ==========================================
    CORNEA_ID = 3            # آیدی قرنیه
    TOOL_IDS = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]  # آیدی ابزارها
    MULTIPLIER = 8           # ضریب تولید
    
    datasets = ['LMM', 'cadis', 'cataract-1k']
    
    # راه‌اندازی کلاس اصلی
    compositor = AdvancedHybridCompositor(CORNEA_ID, TOOL_IDS, out_dir=OUTPUT_DIR)

    # ==========================================
    # ۳. خواندن فایل‌ها (این بخش به صورت خودکار آدرس‌ها را می‌سازد)
    # ==========================================

    # source_pools = {}
    # for ds in datasets:
    #     # جستجو در مسیر: SOURCES_DIR / نام_دیتاست / images / فایل‌های jpg
    #     images = glob.glob(os.path.join(SOURCES_DIR, ds, 'images', '*.jpg')) 
    #     source_pools[ds] = images

    # ==========================================
    # ۳. خواندن فایل‌ها و دسته‌بندی بر اساس کلاس ابزار (Class-Aware Indexing)
    # ==========================================
    source_pools = {ds: {t_id: [] for t_id in TOOL_IDS} for ds in datasets}

    for ds in datasets:
        images = glob.glob(os.path.join(SOURCES_DIR, ds, 'images', '*.jpg'))
        for img_path in images:
            txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            if not os.path.exists(txt_path):
                continue
                
            # خواندن کلاس‌های موجود در فایل txt و "شمارش" تعداد آن‌ها
            with open(txt_path, 'r') as f:
                class_counts = {}
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts:
                        c_id = int(parts[0])
                        class_counts[c_id] = class_counts.get(c_id, 0) + 1
            
            # بررسی و اضافه‌کردن عکس به استخرِ هر ابزار بر اساس شروط شما
            for tool_id in TOOL_IDS:
                if tool_id in class_counts:
                    
                    # ==========================================
                    # شرط اختصاصی برای ابزار کلاس 5
                    # ==========================================
                    if tool_id == 5:
                        # فقط در صورتی عکس را اضافه کن که حداقل 2 تا از این ابزار در عکس باشد
                        if class_counts[tool_id] >= 2:
                            source_pools[ds][tool_id].append(img_path)
                            
                    # برای سایر ابزارها (کلاس‌های غیر از 5)
                    else:
                        # همان یک دانه ابزار هم برای اضافه شدن عکس کافی است
                        source_pools[ds][tool_id].append(img_path)




    for bg_ds in datasets:
        # جستجو در مسیر: BACKGROUNDS_DIR / نام_دیتاست / images / فایل‌های jpg
        bg_images = glob.glob(os.path.join(BACKGROUNDS_DIR, bg_ds, 'images', '*.jpg'))
        other_datasets = [ds for ds in datasets if ds != bg_ds]
        
        for bg_idx, bg_path in enumerate(bg_images):
            bg_img = cv2.imread(bg_path)
            h, w = bg_img.shape[:2]
            
            # پیدا کردن فایل txt هم‌نام با عکس پس‌زمینه
            bg_txt_path = bg_path.replace('images', 'labels').replace('.jpg', '.txt')
            bg_labels = compositor.read_yolo_label(bg_txt_path, w, h)

            for i in range(MULTIPLIER):
                # src_ds_name = other_datasets[i % 2] if i < 4 else random.choice(other_datasets)
                # src_img_path = random.choice(source_pools[src_ds_name])
                
                # # پیدا کردن فایل txt هم‌نام با عکس سورس
                # src_txt_path = src_img_path.replace('images', 'labels').replace('.jpg', '.txt')


                # انتخاب دیتاست سورس (همان منطق قبلی برای توزیع برابر دیتاست‌ها)
                src_ds_name = other_datasets[i % 2] if i < 4 else random.choice(other_datasets)
                
                # ==========================================
                # منطق جدید: انتخاب عادلانه ابزارها
                # ==========================================
                # بررسی اینکه این دیتاست اصلاً کدام ابزارها را درون خود دارد
                available_tools = [t_id for t_id in TOOL_IDS if len(source_pools[src_ds_name][t_id]) > 0]
                
                if not available_tools:
                    continue # اگر دیتاست هیچ ابزاری نداشت (که بعید است)، رد شو
                    
                # 1. ابتدا یک کلاس ابزار را با شانس کاملاً مساوی انتخاب می‌کنیم
                selected_tool_id = random.choice(available_tools)
                
                # 2. حالا از بین عکس‌هایی که قطعاً این ابزار را دارند، یکی را رندوم برمی‌داریم
                src_img_path = random.choice(source_pools[src_ds_name][selected_tool_id])
                
                src_txt_path = src_img_path.replace('images', 'labels').replace('.jpg', '.txt')
                # ... ادامه کدهای قبلی برای ترکیب و ذخیره ..
                
                src_img = cv2.imread(src_img_path)
                src_labels = compositor.read_yolo_label(src_txt_path, w, h)

                # مراحل ترکیب
                tool_data = compositor.transform_tools(src_img, src_labels, bg_labels, bg_img)
                if tool_data[0] is None: 
                    continue
                
                w_tool_img, w_tool_mask, w_tool_labels = tool_data
                # فراخوانی با روش Alpha Feathering (پیشنهاد من برای ابزارهای فلزی)
                # final_img, final_labels = compositor.blend(bg_img, bg_labels, w_tool_img, w_tool_mask, w_tool_labels, use_poisson=False)
                # final_img, final_labels = compositor.blend(bg_img, bg_labels, w_tool_img, w_tool_mask, w_tool_labels, use_poisson=True)
                final_img, final_labels = compositor.blend_gaussian(bg_img, bg_labels, w_tool_img, w_tool_mask, w_tool_labels)

                # استخراج نام دقیق عکس بک‌گراند و سورس (بدون پسوند)
                bg_base_name = os.path.basename(bg_path).replace('.jpg', '')
                src_base_name = os.path.basename(src_img_path).replace('.jpg', '')
                
                # فرمت جدید نام‌گذاری (استفاده از ___ برای جلوگیری از خطای اسپلیت)
                # مثال خروجی: cadis___video1_frame10___aug_0___from___LMM___img_502
                out_name = f"{bg_ds}___{bg_base_name}___aug_{i}___from___{src_ds_name}___{src_base_name}"
                
                cv2.imwrite(os.path.join(compositor.out_img_dir, f"{out_name}.jpg"), final_img)
                compositor.save_yolo_label(os.path.join(compositor.out_label_dir, f"{out_name}.txt"), final_labels, w, h)
                
    print("دیتاست یولو با موفقیت تولید و مختصات Polygons به‌روزرسانی شد!")



def draw_yolo_polygons(image_path, label_path, class_colors):
    """
    خواندن عکس و فایل txt یولو و رسم چندضلعی‌ها با رنگ‌های مشخص
    """
    img = cv2.imread(image_path)
    if img is None:
        # اگر عکس پیدا نشد یک تصویر سیاه برمی‌گرداند
        return np.zeros((500, 500, 3), dtype=np.uint8) 
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                class_id = int(parts[0])
                # انتخاب رنگ بر اساس کلاس (اگر در دیکشنری نبود، سفید در نظر می‌گیرد)
                color = class_colors.get(class_id, (255, 255, 255))
                
                # تبدیل مختصات نرمال‌شده به پیکسلی
                poly_norm = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                poly_abs = np.int32(poly_norm * [w, h])
                
                # رسم خطوط دور (Border) ضخیم
                # cv2.polylines(img, [poly_abs], isClosed=True, color=color, thickness=1)
                
                # ایجاد افکت پر کردن نیمه‌شفاف (Transparent Fill)
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly_abs], color)
                cv2.addWeighted(overlay, 0.0, img, 1.0, 0, img)
                
    return img



def plot_yolo_random_results(sources_dir, backgrounds_dir, output_dir):
    """
    نمایش متناظر: ستون اول (پس‌زمینه هدف)، ستون دوم (سورس واقعی ابزار)، ستون سوم (نتیجه ترکیب)
    """
    aug_images = glob.glob(os.path.join(output_dir, 'images', '*.jpg'))
    if len(aug_images) < 3:
        print("تعداد عکس‌های تولید شده در پوشه خروجی برای پلات کردن کافی نیست!")
        return

    CLASS_COLORS = {
        0: (0, 255, 0),      # قرنیه -> سبز
        1: (255, 0, 0),      # ابزار 1 -> قرمز
        2: (0, 0, 255),      # ابزار 2 -> آبی
        3: (255, 255, 0),    # ابزار 3 -> زرد
        4: (255, 0, 255)     # ابزار 4 -> بنفش
    }

    selected_paths = random.sample(aug_images, 3)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for row, aug_img_path in enumerate(selected_paths):
        aug_txt_path = aug_img_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        # شکستن نام فایل برای پیدا کردن اجزای سازنده
        # فرمت: bgDS ___ bgBaseName ___ aug_i ___ from ___ srcDS ___ srcBaseName
        filename = os.path.basename(aug_img_path).replace('.jpg', '')
        parts = filename.split('___')
        
        bg_ds = parts[0]
        bg_base_name = parts[1]
        src_ds = parts[4]
        src_base_name = parts[5]
        
        # پیدا کردن آدرس دقیق عکس و لیبل بک‌گراند
        bg_img_path = os.path.join(backgrounds_dir, bg_ds, 'images', f'{bg_base_name}.jpg')
        bg_txt_path = os.path.join(backgrounds_dir, bg_ds, 'labels', f'{bg_base_name}.txt')
        
        # پیدا کردن آدرس دقیق عکس و لیبل سورس (منطبق با ستون اول و سوم)
        src_img_path = os.path.join(sources_dir, src_ds, 'images', f'{src_base_name}.jpg')
        src_txt_path = os.path.join(sources_dir, src_ds, 'labels', f'{src_base_name}.txt')

        # رسم پلی‌گان‌ها
        bg_vis = draw_yolo_polygons(bg_img_path, bg_txt_path, CLASS_COLORS)
        src_vis = draw_yolo_polygons(src_img_path, src_txt_path, CLASS_COLORS)
        aug_vis = draw_yolo_polygons(aug_img_path, aug_txt_path, CLASS_COLORS)

        # پلات در ستون اول (بک‌گراند)
        axes[row, 0].imshow(bg_vis)
        axes[row, 0].set_title(f"Target Background ({bg_ds})\nFile: {bg_base_name}", fontsize=11)
        axes[row, 0].axis('off')

        # پلات در ستون دوم (سورس دقیق)
        axes[row, 1].imshow(src_vis)
        axes[row, 1].set_title(f"Exact Source Image ({src_ds})\nFile: {src_base_name}", fontsize=11)
        axes[row, 1].axis('off')

        # پلات در ستون سوم (خروجی)
        axes[row, 2].imshow(aug_vis)
        axes[row, 2].set_title(f"Clinically Composited Result\nMapped to Target Cornea", fontsize=11, color='darkred')
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.show()

def generate_tool_only_dataset(sources_dir, output_dir, tool_ids, total_count=2000, natural_bgs_dir=None):
    """
    تولید دیتاست جدید شامل فقط ابزارها روی پس‌زمینه‌های تصادفی یا رنگی.
    """
    out_img_dir = os.path.join(output_dir, 'images')
    out_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    datasets = ['LMM', 'cadis', 'cataract-1k']
    count_per_ds = total_count // len(datasets) # تقسیم مساوی بین سه دیتاست (حدود 666 عکس برای هر کدام)

    # برای استفاده از توابع خواندن و نوشتن یولو، یک نمونه از کلاس شما می‌سازیم 
    # (آیدی قرنیه مهم نیست چون اینجا فقط ابزار را می‌خواهیم)
    compositor = YoloClinicallyAwareCompositor(cornea_id=3, tool_ids=tool_ids, out_dir=output_dir)

    # لود کردن مسیر پس‌زمینه‌های طبیعت (در صورت وجود)
    natural_bgs = []
    if natural_bgs_dir and os.path.exists(natural_bgs_dir):
        natural_bgs = glob.glob(os.path.join(natural_bgs_dir, '*.jpg'))
        if len(natural_bgs) == 0:
            print("اخطار: عکسی در پوشه طبیعت پیدا نشد. از رنگ‌های تصادفی استفاده می‌شود.")

    for ds in datasets:
        # 1. پیدا کردن تمام عکس‌هایی که حداقل یک ابزار در آن‌ها وجود دارد
        ds_images = glob.glob(os.path.join(sources_dir, ds, 'images', '*.jpg'))
        valid_images = []
        for img_path in ds_images:
            txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            if not os.path.exists(txt_path): continue
            
            # خواندن سریع فایل برای بررسی وجود ابزار
            with open(txt_path, 'r') as f:
                content = f.read()
                # اگر هرکدام از آیدی‌های ابزار در این فایل تکست بود، عکس معتبر است
                if any(str(tid) in content for tid in tool_ids):
                    valid_images.append(img_path)

        if not valid_images:
            print(f"هیچ ابزاری در دیتاست {ds} پیدا نشد!")
            continue

        # 2. انتخاب تصادفی عکس‌ها به تعداد سهمیه این دیتاست (با قابلیت تکرار در صورت نیاز)
        sampled_images = random.choices(valid_images, k=count_per_ds)

        for i, img_path in enumerate(sampled_images):
            txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            labels = compositor.read_yolo_label(txt_path, w, h)
            
            # فیلتر کردن لیبل‌ها: فقط ابزارها را نگه می‌داریم
            tool_labels = [lb for lb in labels if lb['class_id'] in tool_ids]
            if not tool_labels: continue

            # ساخت ماسک ابزارها
            tool_mask = np.zeros((h, w), dtype=np.uint8)
            tool_polys = [lb['polygon'] for lb in tool_labels]
            cv2.fillPoly(tool_mask, tool_polys, 255)

            # استخراج پیکسلی ابزار از عکس اصلی
            tool_img = cv2.bitwise_and(img, img, mask=tool_mask)

            # ==========================================
            # تولید بک‌گراند (طبیعت یا رنگ تصادفی)
            # ==========================================
            if natural_bgs:
                # انتخاب یک عکس رندوم از طبیعت و تغییر سایز آن به ابعاد عکس ما (540x540)
                bg_path = random.choice(natural_bgs)
                bg_img = cv2.imread(bg_path)
                bg_img = cv2.resize(bg_img, (w, h))
            else:
                # تولید یک رنگ کاملاً تصادفی برای کل پس‌زمینه (مثلا یک بار بنفش، یک بار سبز ملایم و...)
                random_color = [random.randint(0, 255) for _ in range(3)]
                bg_img = np.full((h, w, 3), random_color, dtype=np.uint8)

            # ==========================================
            # ترکیب (Blending) با لبه‌های نرم برای جلوگیری از مصنوعی شدن
            # ==========================================
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(tool_mask, kernel, iterations=1)
            soft_mask = cv2.GaussianBlur(eroded_mask, (5, 5), 0) / 255.0
            soft_mask = np.expand_dims(soft_mask, axis=-1)

            final_img = (bg_img * (1 - soft_mask)) + (tool_img * soft_mask)
            final_img = np.clip(final_img, 0, 255).astype(np.uint8)

            # ==========================================
            # ذخیره عکس و فایل txt یولو
            # ==========================================
            base_name = os.path.basename(img_path).replace('.jpg', '')
            out_name = f"toolonly_{ds}_{base_name}_{i}"
            
            cv2.imwrite(os.path.join(out_img_dir, out_name + '.jpg'), final_img)
            # ذخیره فایل تکست که حالا "فقط" حاوی مختصات ابزار است (بدون قرنیه)
            compositor.save_yolo_label(os.path.join(out_label_dir, out_name + '.txt'), tool_labels, w, h)
            
    print(f"دیتاست ابزار-خالص (شامل {total_count} تصویر) با موفقیت تولید شد!")


if __name__ == "__main__":
    # آدرس‌های خود را اینجا قرار دهید (r قبل از استرینگ فراموش نشود)

    SOURCES_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\sources"           # پوشه‌ای که عکس‌های دارای ابزار در آن هستند
    BACKGROUNDS_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\backgrounds"   # پوشه‌ای که پس‌زمینه‌های خالص در آن هستند
    OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\augmented_dataset_method2"  # پوشه‌ای که می‌خواهید خروجی‌ها آنجا ذخیره شوند

    TOOL_ONLY_OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\tool_only_dataset"
    NATURE_BACKGROUNDS_DIR = None # یا مثلاً r"C:\path\to\nature\images"
    TOOL_IDS = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]


    # 1. ابتدا دیتاست را تولید می‌کنیم (اگر قبلاً تولید کرده‌اید، می‌توانید این خط را کامنت کنید)
    # generate_yolo_dataset(SOURCES_DIR, BACKGROUNDS_DIR, OUTPUT_DIR)


    # ---------------------------------------------------------
    # 2. تولید دیتاست جدید (فقط ابزارها) روی پس زمینه رندوم
    # ---------------------------------------------------------
    print("در حال تولید دیتاست ابزارهای ایزوله...")
    generate_tool_only_dataset(
        sources_dir=SOURCES_DIR, 
        output_dir=TOOL_ONLY_OUTPUT_DIR, 
        tool_ids=TOOL_IDS, 
        total_count=2000, 
        natural_bgs_dir=NATURE_BACKGROUNDS_DIR
    )

    # 2. در نهایت نتایج را با آدرس‌های درست پلات می‌کنیم
    plot_yolo_random_results(
        sources_dir=SOURCES_DIR, 
        backgrounds_dir=BACKGROUNDS_DIR, 
        output_dir=OUTPUT_DIR
    )


      
