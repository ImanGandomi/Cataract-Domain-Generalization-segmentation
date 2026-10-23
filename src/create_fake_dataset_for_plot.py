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
        
        # در اینجا فقط لبه‌های ماسک را کمی نرم می‌کنیم تا ابزار بریدگی پیکسلی (دندانه دندانه) نداشته باشد
        # اما تغییری در رنگ خود ابزار ایجاد نمی‌شود.
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(warped_tool_mask, kernel, iterations=1)
        soft_mask = cv2.GaussianBlur(eroded_mask, (7, 7), 0) / 255.0
        soft_mask = np.expand_dims(soft_mask, axis=-1)

        # ==========================================
        # تغییر مهم: حذف محاسبات تغییر رنگ (Color Matching)
        # ==========================================
        # تصویر ابزار را بدون هیچ ضریب و تغییری به صورت مستقیم استفاده می‌کنیم
        adjusted_tool = warped_tool_img.astype(np.float32)

        # ترکیب مستقیم پیکسل‌های ابزار و پس‌زمینه
        final_img = (bg_img * (1 - soft_mask)) + (adjusted_tool * soft_mask)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        final_labels = bg_labels.copy()
        final_labels.extend(warped_tool_labels)

        return final_img, final_labels

# ==========================================
# بخش تولید دیتاست یولو
# ==========================================





def generate_100_pair_dataset(dataset1_dir, dataset2_dir, output_dir, cornea_id, tool_ids):
    """
    ۱۰۰ عکس از dataset1 (دارای ابزار) را با ۱۰۰ عکس از dataset2 (پس‌زمینه) ترکیب می‌کند.
    """
    # ایجاد کلاس اصلی
    compositor = YoloClinicallyAwareCompositor(cornea_id, tool_ids, out_dir=output_dir)

    # گرفتن لیست تمام تصاویر
    src_images = glob.glob(os.path.join(dataset1_dir, 'images', '*.jpg'))
    bg_images = glob.glob(os.path.join(dataset2_dir, 'images', '*.jpg'))

    # اطمینان از اینکه حداقل ۱۰۰ تصویر داریم
    if len(src_images) < 100 or len(bg_images) < 100:
        print(f"خطا: تعداد تصاویر کافی نیست. سورس: {len(src_images)}، پس‌زمینه: {len(bg_images)}")
        return

    # انتخاب ۱۰۰ تای رندوم
    selected_src = random.sample(src_images, 100)
    selected_bg = random.sample(bg_images, 100)

    print(f"شروع ترکیب {len(selected_src)} جفت تصویر...")

    for i, (src_path, bg_path) in enumerate(zip(selected_src, selected_bg)):
        # خواندن تصاویر
        bg_img = cv2.imread(bg_path)
        src_img = cv2.imread(src_path)
        
        h, w = bg_img.shape[:2]

        # خواندن لیبل‌ها
        bg_txt_path = bg_path.replace('images', 'labels').replace('.jpg', '.txt')
        src_txt_path = src_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        bg_labels = compositor.read_yolo_label(bg_txt_path, w, h)
        src_labels = compositor.read_yolo_label(src_txt_path, w, h)

        # انجام عملیات کامپوزیت
        tool_data = compositor.transform_tools(src_img, src_labels, bg_labels, bg_img)
        if tool_data[0] is None: 
            print(f"رد کردن {i}: ابزار مناسبی پیدا نشد.")
            continue
        
        w_tool_img, w_tool_mask, w_tool_labels = tool_data
        
        # ترکیب نهایی
        final_img, final_labels = compositor.blend(bg_img, bg_labels, w_tool_img, w_tool_mask, w_tool_labels, use_poisson=False)

        # نام‌گذاری خروجی
        bg_name = os.path.basename(bg_path).replace('.jpg', '')
        src_name = os.path.basename(src_path).replace('.jpg', '')

        # استخراج نام فایل سورس بدون پسوند
        out_name = os.path.basename(src_path).replace('.jpg', '')
        
        # ذخیره تصویر و لیبل با نام دقیقاً مشابه سورس
        cv2.imwrite(os.path.join(compositor.out_img_dir, f"{out_name}.jpg"), final_img)
        compositor.save_yolo_label(os.path.join(compositor.out_label_dir, f"{out_name}.txt"), final_labels, w, h)

    print(f"کار تمام شد! ۱۰۰ تصویر در {output_dir} ذخیره شدند.")

    print(f"کار تمام شد! ۱۰۰ تصویر در {output_dir} ذخیره شدند.")



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
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                
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



if __name__ == "__main__":
    # آدرس دیتاست‌ها
    DATASET1_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\new\txt\padded_splited_data_114_0_custom\test" # دیتاست دارای ابزار
    DATASET2_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\backgrounds\cadis" # دیتاست پس‌زمینه
    OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\for_plot\1k_on_cadis"

    CORNEA_ID = 3
    TOOL_IDS = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]

    # فراخوانی تابع جدید
    generate_100_pair_dataset(DATASET1_DIR, DATASET2_DIR, OUTPUT_DIR, CORNEA_ID, TOOL_IDS)

    # پلات کردن نتایج
    plot_yolo_random_results(DATASET1_DIR, DATASET2_DIR, OUTPUT_DIR)