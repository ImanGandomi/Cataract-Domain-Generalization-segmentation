import cv2
import numpy as np
import os

class YoloClinicallyAwareCompositor:
    def __init__(self, cornea_id, tool_ids):
        self.cornea_id = cornea_id
        self.tool_ids = tool_ids

    def read_yolo_label(self, txt_path, img_w, img_h):
        """ خواندن فایل txt یولو و تبدیل مختصات نرمال به پیکسلی """
        labels = []
        if not os.path.exists(txt_path):
            return labels
            
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                class_id = int(parts[0])
                poly_norm = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                poly_abs = poly_norm * np.array([img_w, img_h])
                labels.append({"class_id": class_id, "polygon": np.int32(poly_abs)})
        return labels

    def save_yolo_label(self, txt_path, labels, img_w, img_h):
        """ ذخیره لیبل‌های جدید در قالب فایل txt یولو """
        with open(txt_path, 'w') as f:
            for label in labels:
                class_id = label['class_id']
                poly_abs = label['polygon']
                poly_norm = poly_abs.astype(np.float32) / np.array([img_w, img_h])
                poly_norm = np.clip(poly_norm, 0.0, 1.0)
                poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly_norm])
                f.write(f"{class_id} {poly_str}\n")

    def get_valid_y_range(self, img):
        """ پیدا کردن محدوده بافت واقعی (بدون پدینگ‌های خاکستری 114 بالا و پایین) """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray.astype(np.float32) - 114.0)
        valid_rows = np.where(np.mean(diff, axis=1) > 5.0)[0]
        if len(valid_rows) > 0:
            return valid_rows[0], valid_rows[-1]
        return 0, img.shape[0]

    def get_cornea_properties(self, labels, img_w, img_h):
        """ پیدا کردن مرکز و مساحت قرنیه از روی پلی‌گان‌ها """
        cornea_polys = [lb['polygon'] for lb in labels if lb['class_id'] == self.cornea_id]
        if not cornea_polys:
            return None, None
        
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
            print("خطا: هیچ ابزاری در تصویر سورس یافت نشد!")
            return None, None, None

        valid_y_min, valid_y_max = self.get_valid_y_range(bg_img)

        global_min_y, global_max_y = float('inf'), float('-inf')
        temp_polys = []

        for lb in tool_labels:
            poly = lb['polygon'].astype(np.float32)
            new_poly = np.zeros_like(poly)
            new_poly[:, 0] = poly[:, 0] * scale + tx
            new_poly[:, 1] = poly[:, 1] * scale + ty
            temp_polys.append(new_poly)
            global_min_y = min(global_min_y, np.min(new_poly[:, 1]))
            global_max_y = max(global_max_y, np.max(new_poly[:, 1]))

        group_height = global_max_y - global_min_y
        valid_height = valid_y_max - valid_y_min

        if group_height > valid_height:
            print("خطا: قد ابزارها از کل ارتفاع بافت چشم بزرگتر است!")
            return None, None, None

        shift_y = 0
        if global_min_y < valid_y_min:
            shift_y = valid_y_min - global_min_y 
        elif global_max_y > valid_y_max:
            shift_y = valid_y_max - global_max_y 

        ty += shift_y

        warped_tool_labels = []
        for i, lb in enumerate(tool_labels):
            final_poly = temp_polys[i]
            final_poly[:, 1] += shift_y
            warped_tool_labels.append({"class_id": lb['class_id'], "polygon": final_poly.astype(np.int32)})

        M = np.float32([[scale, 0, tx], [0, scale, ty]])

        tool_mask = np.zeros((h, w), dtype=np.uint8)
        tool_polys = [lb['polygon'] for lb in tool_labels]
        cv2.fillPoly(tool_mask, tool_polys, 255)

        tool_img = cv2.bitwise_and(src_img, src_img, mask=tool_mask)
        warped_tool_img = cv2.warpAffine(tool_img, M, (w, h), flags=cv2.INTER_LINEAR)
        warped_tool_mask = cv2.warpAffine(tool_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        return warped_tool_img, warped_tool_mask, warped_tool_labels

    def blend(self, bg_img, bg_labels, warped_tool_img, warped_tool_mask, warped_tool_labels, use_feathering=True, use_color_adjustment=True):
        valid_y_min, valid_y_max = self.get_valid_y_range(bg_img)

        # ==========================================
        # تاگل بخش اول: Alpha Feathering
        # ==========================================
        if use_feathering:
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(warped_tool_mask, kernel, iterations=1)
            soft_mask = cv2.GaussianBlur(eroded_mask, (7, 7), 0) / 255.0
            soft_mask = np.expand_dims(soft_mask, axis=-1)
        else:
            # اگر فدرینگ غیرفعال باشد، ماسک معمولی با لبه‌های تیز استفاده می‌شود
            soft_mask = (warped_tool_mask / 255.0)
            soft_mask = np.expand_dims(soft_mask, axis=-1)

        # ==========================================
        # تاگل بخش دوم: Color Ratio Adjustment
        # ==========================================
        if use_color_adjustment:
            valid_bg_region = bg_img[valid_y_min:valid_y_max, :]
            valid_mask_region = warped_tool_mask[valid_y_min:valid_y_max, :]
            pure_bg_pixels = valid_bg_region[valid_mask_region == 0]

            bg_mean = np.mean(pure_bg_pixels) if len(pure_bg_pixels) > 0 else 127.0
            tool_pixels = warped_tool_img[warped_tool_mask > 0]
            tool_mean = np.mean(tool_pixels) if len(tool_pixels) > 0 else 1.0
            
            if tool_mean > 0:
                color_ratio = (bg_mean / tool_mean) * 0.3 + 0.7 
                adjusted_tool = np.clip(warped_tool_img * color_ratio, 0, 255).astype(np.float32)
            else:
                adjusted_tool = warped_tool_img.astype(np.float32)
        else:
            # اگر تغییر رنگ غیرفعال باشد، از خود تصویر ابزار اصلی بدون دستکاری نوری استفاده می‌شود
            adjusted_tool = warped_tool_img.astype(np.float32)

        # ترکیب نهایی تصویر
        final_img = (bg_img * (1 - soft_mask)) + (adjusted_tool * soft_mask)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        final_labels = bg_labels.copy()
        final_labels.extend(warped_tool_labels)

        return final_img, final_labels


def composite_single_pair(bg_img_path, bg_txt_path, src_img_path, src_txt_path, out_img_path, out_label_path, use_feathering=True, use_color_adjustment=True):
    """ تابع تک جفتی مجهز به کلیدهای کنترلی افکت‌ها """
    CORNEA_ID = 3
    TOOL_IDS = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]

    compositor = YoloClinicallyAwareCompositor(CORNEA_ID, TOOL_IDS)

    bg_img = cv2.imread(bg_img_path)
    src_img = cv2.imread(src_img_path)

    if bg_img is None or src_img is None:
        print("خطا در بارگذاری تصاویر! لطفاً مسیرها را چک کنید.")
        return

    h_bg, w_bg = bg_img.shape[:2]

    bg_labels = compositor.read_yolo_label(bg_txt_path, w_bg, h_bg)
    src_labels = compositor.read_yolo_label(src_txt_path, w_bg, h_bg)

    tool_data = compositor.transform_tools(src_img, src_labels, bg_labels, bg_img)
    if tool_data[0] is None:
        print("فرآیند جابجایی ابزار با خطا مواجه شد.")
        return

    w_tool_img, w_tool_mask, w_tool_labels = tool_data

    # پاس دادن تاگل‌ها به تابع blend
    final_img, final_labels = compositor.blend(
        bg_img, bg_labels, w_tool_img, w_tool_mask, w_tool_labels,
        use_feathering=use_feathering,
        use_color_adjustment=use_color_adjustment
    )

    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_label_path), exist_ok=True)

    cv2.imwrite(out_img_path, final_img)
    compositor.save_yolo_label(out_label_path, final_labels, w_bg, h_bg)
    print(f"تصویر خروجی ذخیره شد. فدرینگ: {use_feathering} | اصلاح رنگ: {use_color_adjustment}")


# =========================================================
# بخش تنظیمات و اجرا
# =========================================================
if __name__ == "__main__":

    BACKGROUND_IMAGE = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\one_image_create_fake_dataset\Video25_frame006610_png_jpg.rf.673c39ad87b4b75231388f62a0f419d8.jpg"
    BACKGROUND_LABEL = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\one_image_create_fake_dataset\Video25_frame006610_png_jpg.rf.673c39ad87b4b75231388f62a0f419d8.txt"

    SOURCE_IMAGE = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\one_image_create_fake_dataset\case5304_11_png.rf.21a54f0f3a08d7239c419c01e2a9ca1f.jpg"
    SOURCE_LABEL = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\one_image_create_fake_dataset\case5304_11_png.rf.21a54f0f3a08d7239c419c01e2a9ca1f.txt"

    OUTPUT_IMAGE = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\one_image_create_fake_dataset\aa2\result_image_FEATHERING_yes7_COLOR_yes.jpg"
    OUTPUT_LABEL = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\plot_for_paper\one_image_create_fake_dataset\aa2\result_label_FEATHERING_yes7_COLOR_yes.txt"

    # -----------------------------------------------------
    # تاگل‌ها را اینجا تنظیم کنید (True برای فعال و False برای غیرفعال)
    # -----------------------------------------------------
    FEATHERING_TOGGLE = True          # کنترل نرم شدن لبه‌ها
    COLOR_ADJUSTMENT_TOGGLE = True     # کنترل تغییر اتوماتیک رنگ/نور ابزار

    composite_single_pair(
        bg_img_path=BACKGROUND_IMAGE,
        bg_txt_path=BACKGROUND_LABEL,
        src_img_path=SOURCE_IMAGE,
        src_txt_path=SOURCE_LABEL,
        out_img_path=OUTPUT_IMAGE,
        out_label_path=OUTPUT_LABEL,
        use_feathering=FEATHERING_TOGGLE,
        use_color_adjustment=COLOR_ADJUSTMENT_TOGGLE
    )