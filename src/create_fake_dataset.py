import cv2
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt

class YoloClinicallyAwareCompositor:
    def __init__(self, cornea_id, tool_ids, out_dir):
        self.cornea_id = cornea_id
        self.tool_ids = tool_ids
        
        self.out_img_dir = os.path.join(out_dir, 'images')
        self.out_label_dir = os.path.join(out_dir, 'labels')
        os.makedirs(self.out_img_dir, exist_ok=True)
        os.makedirs(self.out_label_dir, exist_ok=True)

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




    # def get_valid_y_range(self, img):                                                         # for black padding
    #     """ پیدا کردن محدوده بافت واقعی (بدون پدینگ‌های سیاه بالا و پایین) """
    #     # تبدیل به خاکستری برای پیدا کردن پیکسل‌های روشن
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #     # پیدا کردن سطرهایی که مجموع روشنایی آن‌ها بیشتر از صفر (یا یک مقدار حداقل) است
    #     # عدد 10 برای این است که نویزهای احتمالی در پدینگ سیاه نادیده گرفته شوند
    #     valid_rows = np.where(np.sum(gray, axis=1) > 10)[0]
        
    #     if len(valid_rows) > 0:
    #         y_min = valid_rows[0]
    #         y_max = valid_rows[-1]
    #         return y_min, y_max
    #     return 0, img.shape[0] # اگر کلا سیاه بود (که محال است)، کل ارتفاع را برگردان
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
        """ پیدا کردن مرکز و مساحت قرنیه از روی پلی‌گان‌ها """
        cornea_polys = [lb['polygon'] for lb in labels if lb['class_id'] == self.cornea_id]
        if not cornea_polys:
            return None, None
        
        # برای محاسبه مساحت و مرکز، موقتاً یک ماسک خالی می‌سازیم
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, cornea_polys, 1)
        
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = M["m00"]
            return (cx, cy), area
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


    # def blend(self, bg_img, bg_labels, warped_tool_img, warped_tool_mask, warped_tool_labels):
    #     """ ترکیب نهایی عکس و لیست لیبل‌ها """
    #     # بلور کردن مرزهای ماسک ابزار برای چسباندن طبیعی
    #     soft_mask = cv2.GaussianBlur(warped_tool_mask, (5, 5), 0) / 255.0
    #     soft_mask = np.expand_dims(soft_mask, axis=-1)

    #     final_img = (bg_img * (1 - soft_mask)) + (warped_tool_img * soft_mask)
    #     final_img = np.clip(final_img, 0, 255).astype(np.uint8)

    #     # ترکیب لیبل‌های بک‌گراند (مثل قرنیه) با لیبل ابزارهای جدید که تغییر مقیاس داده‌اند
    #     final_labels = bg_labels.copy()
    #     final_labels.extend(warped_tool_labels)

    #     return final_img, final_labels
    def blend(self, bg_img, bg_labels, warped_tool_img, warped_tool_mask, warped_tool_labels, use_poisson=False):
        
        # پیدا کردن محدوده بافت چشم برای جلوگیری از محاسبه خاکستری‌ها
        valid_y_min, valid_y_max = self.get_valid_y_range(bg_img)

        if not use_poisson:
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(warped_tool_mask, kernel, iterations=1)
            soft_mask = cv2.GaussianBlur(eroded_mask, (7, 7), 0) / 255.0
            soft_mask = np.expand_dims(soft_mask, axis=-1)

            # ==========================================
            # اصلاح تطبیق رنگ: فیلتر کردن پدینگ‌های خاکستری
            # ==========================================
            # فقط ناحیه‌ای از عکس که بافت چشم است را جدا می‌کنیم
            valid_bg_region = bg_img[valid_y_min:valid_y_max, :]
            valid_mask_region = warped_tool_mask[valid_y_min:valid_y_max, :]

            # حالا پیکسل‌هایی را می‌گیریم که "درون چشم هستند" و "ابزار روی آن‌ها نیست"
            pure_bg_pixels = valid_bg_region[valid_mask_region == 0]

            if len(pure_bg_pixels) > 0:
                bg_mean = np.mean(pure_bg_pixels)
            else:
                bg_mean = 127.0 # مقدار پیش‌فرض امن

            tool_pixels = warped_tool_img[warped_tool_mask > 0]
            tool_mean = np.mean(tool_pixels) if len(tool_pixels) > 0 else 1.0
            
            if tool_mean > 0:
                color_ratio = (bg_mean / tool_mean) * 0.3 + 0.7 
                adjusted_tool = np.clip(warped_tool_img * color_ratio, 0, 255).astype(np.float32)
            else:
                adjusted_tool = warped_tool_img.astype(np.float32)

            final_img = (bg_img * (1 - soft_mask)) + (adjusted_tool * soft_mask)
            final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        else:
            # روش پواسون (در صورت نیاز)
            x, y, w_rect, h_rect = cv2.boundingRect(warped_tool_mask)
            if w_rect > 0 and h_rect > 0:
                center = (x + w_rect // 2, y + h_rect // 2)
                try:
                    final_img = cv2.seamlessClone(warped_tool_img, bg_img, warped_tool_mask, center, cv2.NORMAL_CLONE)
                except:
                    final_img = np.where(np.expand_dims(warped_tool_mask, -1) > 0, warped_tool_img, bg_img)
            else:
                final_img = np.where(np.expand_dims(warped_tool_mask, -1) > 0, warped_tool_img, bg_img)

        final_labels = bg_labels.copy()
        final_labels.extend(warped_tool_labels)

        return final_img, final_labels

# ==========================================
# بخش تولید دیتاست یولو
# ==========================================
def generate_yolo_dataset():
    # ==========================================
    # ۱. آدرس‌دهی مسیرهای اصلی (این ۳ خط را تغییر دهید)
    # ==========================================
    
    SOURCES_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\sources"           # پوشه‌ای که عکس‌های دارای ابزار در آن هستند
    BACKGROUNDS_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\backgrounds"   # پوشه‌ای که پس‌زمینه‌های خالص در آن هستند
    OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\augmented_dataset2"  # پوشه‌ای که می‌خواهید خروجی‌ها آنجا ذخیره شوند

    # ==========================================
    # ۲. تنظیمات کلاس‌ها (بر اساس فایل classes.yaml شما)
    # ==========================================
    CORNEA_ID = 3            # آیدی قرنیه
    TOOL_IDS = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]  # آیدی ابزارها
    MULTIPLIER = 8           # ضریب تولید
    
    datasets = ['LMM', 'cadis', 'cataract-1k']
    
    # راه‌اندازی کلاس اصلی
    compositor = YoloClinicallyAwareCompositor(CORNEA_ID, TOOL_IDS, out_dir=OUTPUT_DIR)

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
                final_img, final_labels = compositor.blend(bg_img, bg_labels, w_tool_img, w_tool_mask, w_tool_labels, use_poisson=False)
                # final_img, final_labels = compositor.blend(bg_img, bg_labels, w_tool_img, w_tool_mask, w_tool_labels, use_poisson=True)

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
    تولید دیتاست جدید شامل فقط ابزارها روی پس‌زمینه‌های تصادفی، با توزیع کاملاً برابر بین ابزارها و دیتاست‌ها.
    """
    out_img_dir = os.path.join(output_dir, 'images')
    out_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    datasets = ['LMM', 'cadis', 'cataract-1k']
    compositor = YoloClinicallyAwareCompositor(cornea_id=3, tool_ids=tool_ids, out_dir=output_dir)

    # لود کردن پس‌زمینه‌های طبیعت (در صورت وجود)
    natural_bgs = []
    if natural_bgs_dir and os.path.exists(natural_bgs_dir):
        natural_bgs = glob.glob(os.path.join(natural_bgs_dir, '*.jpg'))
    
    # ==========================================
    # 1. دسته‌بندی دقیق عکس‌ها بر اساس کلاس ابزار
    # ==========================================
    source_pools = {ds: {t_id: [] for t_id in tool_ids} for ds in datasets}

    for ds in datasets:
        images = glob.glob(os.path.join(sources_dir, ds, 'images', '*.jpg'))
        for img_path in images:
            txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            if not os.path.exists(txt_path): continue
            
            with open(txt_path, 'r') as f:
                class_counts = {}
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts:
                        c_id = int(parts[0])
                        class_counts[c_id] = class_counts.get(c_id, 0) + 1
            
            for tool_id in tool_ids:
                if tool_id in class_counts:
                    # همان شرط قبلی: برای ابزار 5 حتما باید 2 تا در تصویر باشد
                    if tool_id == 5:
                        if class_counts[tool_id] >= 1:
                            source_pools[ds][tool_id].append(img_path)
                    else:
                        source_pools[ds][tool_id].append(img_path)

    # ==========================================
    # 2. محاسبه سهمیه دقیق هر ابزار
    # ==========================================
    count_per_tool = total_count // len(tool_ids)
    print(f"هدف: تولید {count_per_tool} عکس برای هر کلاسِ ابزار.")

    # ==========================================
    # 3. حلقه تولید متوازن
    # ==========================================
    produced_total = 0

    for tool_id in tool_ids:
        for i in range(count_per_tool):
            # پیدا کردن دیتاست‌هایی که این ابزار را دارند
            available_datasets = [ds for ds in datasets if len(source_pools[ds][tool_id]) > 0]
            
            if not available_datasets:
                print(f"اخطار: ابزار کلاس {tool_id} در هیچ دیتاستی پیدا نشد!")
                break # رفتن به ابزار بعدی

            # انتخاب چرخشی و عادلانه دیتاست (برای رعایت سهم 1/3 هر دیتاست)
            # اگر ابزاری در یک دیتاست نبود، از دیتاست‌های موجود رندوم برمی‌دارد
            if len(available_datasets) == 3:
                selected_ds = available_datasets[i % 3]
            else:
                selected_ds = random.choice(available_datasets)

            # انتخاب یک عکس رندوم از استخرِ این ابزار در دیتاست انتخاب شده
            img_path = random.choice(source_pools[selected_ds][tool_id])
            txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            labels = compositor.read_yolo_label(txt_path, w, h)
            
            # فیلتر لیبل‌ها: فقط و فقط ابزارها (حذف قرنیه و بافت‌ها)
            tool_labels = [lb for lb in labels if lb['class_id'] in tool_ids]
            
            tool_mask = np.zeros((h, w), dtype=np.uint8)
            tool_polys = [lb['polygon'] for lb in tool_labels]
            cv2.fillPoly(tool_mask, tool_polys, 255)

            tool_img = cv2.bitwise_and(img, img, mask=tool_mask)

            # تولید پس‌زمینه جدید (طبیعت یا رنگ رندوم)
            if natural_bgs:
                bg_path = random.choice(natural_bgs)
                bg_img = cv2.imread(bg_path)
                bg_img = cv2.resize(bg_img, (w, h))
            else:
                # تولید رنگ‌های پس‌زمینه کاملا تصادفی و متنوع
                random_color = [random.randint(20, 230) for _ in range(3)]
                bg_img = np.full((h, w, 3), random_color, dtype=np.uint8)

            # ==========================================
            # اعمال ایده هوشمندانه شما: اضافه کردن پدینگ خاکستری به عکس رندوم
            # ==========================================
            # 1. پیدا کردن مرزهای دقیق پدینگ از عکس واقعی (سورس)
            valid_y_min, valid_y_max = compositor.get_valid_y_range(img)
            
            # 2. نقاشی کردن نوار خاکستری (114, 114, 114) در بالا و پایین عکس رندوم
            if valid_y_min > 0:
                bg_img[0:valid_y_min, :] = (114, 114, 114) # پدینگ بالا
            if valid_y_max < h:
                bg_img[valid_y_max:h, :] = (114, 114, 114) # پدینگ پایین

                

            # بلِندینگ و نرم کردن لبه‌های ابزار
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(tool_mask, kernel, iterations=1)
            soft_mask = cv2.GaussianBlur(eroded_mask, (5, 5), 0) / 255.0
            soft_mask = np.expand_dims(soft_mask, axis=-1)

            final_img = (bg_img * (1 - soft_mask)) + (tool_img * soft_mask)
            final_img = np.clip(final_img, 0, 255).astype(np.uint8)

            # نام‌گذاری و ذخیره
            base_name = os.path.basename(img_path).replace('.jpg', '')
            out_name = f"toolonly_targetClass{tool_id}_{selected_ds}_{base_name}_{i}"
            
            cv2.imwrite(os.path.join(out_img_dir, out_name + '.jpg'), final_img)
            compositor.save_yolo_label(os.path.join(out_label_dir, out_name + '.txt'), tool_labels, w, h)
            
            produced_total += 1

    print(f"عملیات موفق! تعداد {produced_total} تصویر کاملاً ایزوله و بالانس‌شده تولید شد.")


if __name__ == "__main__":
    # آدرس‌های خود را اینجا قرار دهید (r قبل از استرینگ فراموش نشود)

    SOURCES_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\sources"           # پوشه‌ای که عکس‌های دارای ابزار در آن هستند
    BACKGROUNDS_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\backgrounds"   # پوشه‌ای که پس‌زمینه‌های خالص در آن هستند
    OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\augmented_dataset"  # پوشه‌ای که می‌خواهید خروجی‌ها آنجا ذخیره شوند

    TOOL_ONLY_OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\tool_only_dataset"
    NATURE_BACKGROUNDS_DIR = None # یا مثلاً r"C:\path\to\nature\images"
    TOOL_IDS = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]

    # 1. ابتدا دیتاست را تولید می‌کنیم (اگر قبلاً تولید کرده‌اید، می‌توانید این خط را کامنت کنید)
    # generate_yolo_dataset()


    # ---------------------------------------------------------
    # 2. تولید دیتاست جدید (فقط ابزارها) روی پس زمینه رندوم
    # ---------------------------------------------------------
    print("در حال تولید دیتاست ابزارهای ایزوله...")
    # generate_tool_only_dataset(
    #     sources_dir=SOURCES_DIR, 
    #     output_dir=TOOL_ONLY_OUTPUT_DIR, 
    #     tool_ids=TOOL_IDS, 
    #     total_count=2000, 
    #     natural_bgs_dir=NATURE_BACKGROUNDS_DIR
    # )


    # 2. در نهایت نتایج را با آدرس‌های درست پلات می‌کنیم
    plot_yolo_random_results(
        sources_dir=SOURCES_DIR, 
        backgrounds_dir=BACKGROUNDS_DIR, 
        output_dir=OUTPUT_DIR
    )


