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
        labels = []
        if not os.path.exists(txt_path): return labels
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
        with open(txt_path, 'w') as f:
            for label in labels:
                class_id = label['class_id']
                poly_norm = label['polygon'].astype(np.float32) / np.array([img_w, img_h])
                poly_norm = np.clip(poly_norm, 0.0, 1.0)
                poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly_norm])
                f.write(f"{class_id} {poly_str}\n")

    def get_valid_y_range(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray.astype(np.float32) - 114.0)
        valid_rows = np.where(np.mean(diff, axis=1) > 5.0)[0]
        if len(valid_rows) > 0:
            return valid_rows[0], valid_rows[-1]
        return 0, img.shape[0]

    def get_cornea_properties(self, labels, img_w, img_h):
        cornea_polys = [lb['polygon'] for lb in labels if lb['class_id'] == self.cornea_id]
        if not cornea_polys: return None, None
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, cornea_polys, 1)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), M["m00"]
        return None, None

    def apply_domain_style(self, bg_img, valid_y_min, valid_y_max):
        img_array = bg_img.copy().astype(np.float32)
        active_region = img_array[valid_y_min:valid_y_max, :, :]
        
        if random.random() > 0.99:
            color_shift = np.random.uniform(-15, 15, 3)
            active_region[:, :, 0] += color_shift[0]
            active_region[:, :, 1] += color_shift[1]
            active_region[:, :, 2] += color_shift[2]
            
        if random.random() > 0.99:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            active_region = active_region * alpha + beta
            
        if random.random() > 0.99:
            blur_kernel = random.choice([3, 5])
            active_region = cv2.GaussianBlur(active_region, (blur_kernel, blur_kernel), 0)

        img_array[valid_y_min:valid_y_max, :, :] = active_region
        return np.clip(img_array, 0, 255).astype(np.uint8)



    def check_mask_collision(self, existing_mask, new_tool_mask, tolerance_ratio=0.08):
        """ 
        بررسی تداخل: به جای صفر مطلق، اجازه می‌دهیم تا 8 درصد ابزار جدید 
        روی ابزارهای دیگر بیفتد.
        """
        intersection = cv2.bitwise_and((existing_mask > 0).astype(np.uint8), (new_tool_mask > 0).astype(np.uint8))
        intersection_area = np.sum(intersection)
        new_tool_area = np.sum(new_tool_mask > 0)
        
        if new_tool_area == 0: 
            return True # اگر ابزار اصلا مساحتی ندارد، رد شود
            
        overlap_ratio = intersection_area / new_tool_area
        return overlap_ratio > tolerance_ratio # اگر تداخل بیشتر از 8% بود، True برگردان (یعنی تداخل غیرمجاز)

    def transform_specific_tool(self, src_img, src_labels, bg_labels, bg_img, target_tool_id, angle, shift_x, shift_y, allow_crop=False):
        h, w = src_img.shape[:2]
        src_center, src_area = self.get_cornea_properties(src_labels, w, h)
        bg_center, bg_area = self.get_cornea_properties(bg_labels, w, h)

        if not src_center or not bg_center:
            return None, None, None

        scale = np.sqrt(bg_area / src_area)
        cx_src, cy_src = src_center
        cx_bg, cy_bg = bg_center

        specific_tool_labels = [lb for lb in src_labels if lb['class_id'] == target_tool_id]
        if not specific_tool_labels:
            return None, None, None

        M1 = np.float32([[1, 0, -cx_src], [0, 1, -cy_src], [0, 0, 1]])
        S = np.float32([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        M2 = np.float32([[1, 0, cx_bg], [0, 1, cy_bg], [0, 0, 1]])
        R_cv2 = cv2.getRotationMatrix2D((cx_bg, cy_bg), angle, 1.0)
        R = np.vstack([R_cv2, [0, 0, 1]])
        M3 = np.float32([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]])

        M_final = M3 @ R @ M2 @ S @ M1
        M_affine = M_final[:2, :]

        valid_y_min, valid_y_max = self.get_valid_y_range(bg_img)
        warped_tool_labels = []
        
        for lb in specific_tool_labels:
            poly = lb['polygon'].astype(np.float32)
            pts = np.hstack([poly, np.ones((len(poly), 1))])
            new_poly = pts.dot(M_affine.T)
            
            # بررسی خروج از کادر
            out_of_bounds = (np.any(new_poly[:, 0] < 0) or np.any(new_poly[:, 0] >= w) or
                             np.any(new_poly[:, 1] < valid_y_min) or np.any(new_poly[:, 1] >= valid_y_max))
            
            # اگر اجازه برش نداریم و ابزار بیرون زده است، کلا رد می‌شود
            if out_of_bounds and not allow_crop:
                return None, None, None
            
            # اگر اجازه برش داریم، نقاط پولیگان را در مرز کادر تصویر و پدینگ‌ها قطع می‌کنیم
            if allow_crop:
                new_poly[:, 0] = np.clip(new_poly[:, 0], 0, w - 1)
                new_poly[:, 1] = np.clip(new_poly[:, 1], valid_y_min, valid_y_max - 1)
                
            warped_tool_labels.append({"class_id": lb['class_id'], "polygon": new_poly.astype(np.int32)})

        tool_mask = np.zeros((h, w), dtype=np.uint8)
        tool_polys = [lb['polygon'] for lb in specific_tool_labels]
        cv2.fillPoly(tool_mask, tool_polys, 255)

        tool_img = cv2.bitwise_and(src_img, src_img, mask=tool_mask)
        warped_tool_img = cv2.warpAffine(tool_img, M_affine, (w, h), flags=cv2.INTER_LINEAR)
        warped_tool_mask = cv2.warpAffine(tool_mask, M_affine, (w, h), flags=cv2.INTER_NEAREST)

        # کات کردن بصری: هر قسمتی از ماسک ابزار که روی پدینگ افتاده را صفر می‌کنیم
        if allow_crop:
            warped_tool_mask[:valid_y_min, :] = 0
            warped_tool_mask[valid_y_max:, :] = 0
            
            # اگر بعد از کات کردن، تقریباً تمام ابزار حذف شده بود (کمتر از 100 پیکسل مساحت داشت)، آن را رد می‌کنیم
            if np.sum(warped_tool_mask > 0) < 100:
                return None, None, None

        return warped_tool_img, warped_tool_mask, warped_tool_labels


    def blend(self, bg_img, bg_labels, warped_tool_img, warped_tool_mask, warped_tool_labels, use_poisson=False):
        valid_y_min, valid_y_max = self.get_valid_y_range(bg_img)

        if not use_poisson:
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(warped_tool_mask, kernel, iterations=1)
            soft_mask = cv2.GaussianBlur(eroded_mask, (7, 7), 0) / 255.0
            soft_mask = np.expand_dims(soft_mask, axis=-1)

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

            final_img = (bg_img * (1 - soft_mask)) + (adjusted_tool * soft_mask)
            final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        final_labels = bg_labels.copy()
        final_labels.extend(warped_tool_labels)
        return final_img, final_labels

# =======================================================================

def draw_yolo_polygons(image_path, label_path, class_colors):
    img = cv2.imread(image_path)
    if img is None: return np.zeros((500, 500, 3), dtype=np.uint8) 
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                class_id = int(parts[0])
                color = class_colors.get(class_id, (255, 255, 255))
                poly_norm = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                poly_abs = np.int32(poly_norm * [w, h])
                
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly_abs], color)
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    return img

def generate_yolo_dataset(compositor, datasets, source_pools, tool_ids, sources_dir):
    generation_history = []
    
    for target_ds in datasets:
        target_images = glob.glob(os.path.join(sources_dir, target_ds, 'images', '*.jpg'))
        other_datasets = [ds for ds in datasets if ds != target_ds]
        
        for target_img_path in target_images:
            target_img = cv2.imread(target_img_path)
            h, w = target_img.shape[:2]
            
            target_txt_path = target_img_path.replace('images', 'labels').replace('.jpg', '.txt')
            target_labels = compositor.read_yolo_label(target_txt_path, w, h)
            
            valid_y_min, valid_y_max = compositor.get_valid_y_range(target_img)
            current_bg_img = compositor.apply_domain_style(target_img, valid_y_min, valid_y_max)
            current_bg_labels = target_labels.copy()
            
            used_source_imgs = []
            used_source_txts = []
            
            present_tools = list(set([lb['class_id'] for lb in target_labels if lb['class_id'] in tool_ids]))
            
            if present_tools:
                existing_tools_mask = np.zeros((h, w), dtype=np.uint8)
                for lb in current_bg_labels:
                    if lb['class_id'] != compositor.cornea_id:
                        cv2.fillPoly(existing_tools_mask, [lb['polygon']], 1)

                for tool_cls in present_tools:
                    # تغییر اصلی اینجاست: حلقه روی هر دیتاست به صورت مجزا اجرا می‌شود
                    for src_ds_name in other_datasets:
                        potential_sources = source_pools[src_ds_name][tool_cls].copy()
                        
                        if not potential_sources: continue
                        random.shuffle(potential_sources)
                        
                        success = False
                        # تلاش برای پیدا کردن یک سورس مناسب فقط از "همین" دیتاست
                        for src_img_path in potential_sources[:3]:
                            if success: break
                                
                            src_txt_path = src_img_path.replace('images', 'labels').replace('.jpg', '.txt')
                            src_img = cv2.imread(src_img_path)
                            src_labels = compositor.read_yolo_label(src_txt_path, w, h)
                            
                            angles = list(range(0, 360, 5))
                            random.shuffle(angles)
                            
                            # ==========================================
                            # فاز ۱: ایده‌آل (چرخش خالص بدون برش)
                            # ==========================================
                            for angle in angles:
                                tool_data = compositor.transform_specific_tool(
                                    src_img, src_labels, current_bg_labels, current_bg_img, target_tool_id=tool_cls, angle=angle, shift_x=0, shift_y=0, allow_crop=False
                                )
                                if tool_data[0] is not None:
                                    w_tool_img, w_tool_mask, w_tool_labels = tool_data
                                    if not compositor.check_mask_collision(existing_tools_mask, w_tool_mask, tolerance_ratio=0.08):
                                        current_bg_img, current_bg_labels = compositor.blend(current_bg_img, current_bg_labels, w_tool_img, w_tool_mask, w_tool_labels)
                                        cv2.fillPoly(existing_tools_mask, [lb['polygon'] for lb in w_tool_labels], 1)
                                        used_source_imgs.append(src_img_path)
                                        used_source_txts.append(src_txt_path)
                                        success = True
                                        break
                                        
                            if success: break
                            
                            # ==========================================
                            # فاز ۲: انعطاف‌پذیر (چرخش + شیفت ملایم بدون برش)
                            # ==========================================
                            for angle in angles[:20]:
                                if success: break
                                for _ in range(3):
                                    if success: break
                                    shift_x, shift_y = random.randint(-40, 40), random.randint(-40, 40)
                                    tool_data = compositor.transform_specific_tool(
                                        src_img, src_labels, current_bg_labels, current_bg_img, target_tool_id=tool_cls, angle=angle, shift_x=shift_x, shift_y=shift_y, allow_crop=False
                                    )
                                    if tool_data[0] is not None:
                                        w_tool_img, w_tool_mask, w_tool_labels = tool_data
                                        if not compositor.check_mask_collision(existing_tools_mask, w_tool_mask, tolerance_ratio=0.08):
                                            current_bg_img, current_bg_labels = compositor.blend(current_bg_img, current_bg_labels, w_tool_img, w_tool_mask, w_tool_labels)
                                            cv2.fillPoly(existing_tools_mask, [lb['polygon'] for lb in w_tool_labels], 1)
                                            used_source_imgs.append(src_img_path)
                                            used_source_txts.append(src_txt_path)
                                            success = True
                                            break

                            # ==========================================
                            # فاز ۳: عملیات نجات (چرخش + شیفت بزرگ + اجازه برش)
                            # ==========================================
                            if not success:
                                for angle in angles[:15]:
                                    if success: break
                                    for _ in range(4): 
                                        if success: break
                                        shift_x, shift_y = random.randint(-120, 120), random.randint(-100, 100)
                                        tool_data = compositor.transform_specific_tool(
                                            src_img, src_labels, current_bg_labels, current_bg_img, target_tool_id=tool_cls, angle=angle, shift_x=shift_x, shift_y=shift_y, allow_crop=True
                                        )
                                        if tool_data[0] is not None:
                                            w_tool_img, w_tool_mask, w_tool_labels = tool_data
                                            if not compositor.check_mask_collision(existing_tools_mask, w_tool_mask, tolerance_ratio=0.15):
                                                current_bg_img, current_bg_labels = compositor.blend(current_bg_img, current_bg_labels, w_tool_img, w_tool_mask, w_tool_labels)
                                                cv2.fillPoly(existing_tools_mask, [lb['polygon'] for lb in w_tool_labels], 1)
                                                used_source_imgs.append(src_img_path)
                                                used_source_txts.append(src_txt_path)
                                                success = True
                                                break

            # 3. ذخیره خروجی
            bg_base_name = os.path.basename(target_img_path).replace('.jpg', '')
            out_name = f"{target_ds}___{bg_base_name}___augmented"
            
            final_img_path = os.path.join(compositor.out_img_dir, f"{out_name}.jpg")
            final_txt_path = os.path.join(compositor.out_label_dir, f"{out_name}.txt")
            
            cv2.imwrite(final_img_path, current_bg_img)
            compositor.save_yolo_label(final_txt_path, current_bg_labels, w, h)
            
            if len(used_source_imgs) > 0:
                generation_history.append({
                    'target_img': target_img_path,
                    'target_txt': target_txt_path,
                    'source_imgs': used_source_imgs,
                    'source_txts': used_source_txts,
                    'final_img': final_img_path,
                    'final_txt': final_txt_path
                })
                
    print("تولید دیتاست انجام شد. حالا از هر دو دیتاست به صورت مجزا ابزار کپی می‌شود!")
    return generation_history

# مابقی کد پلات کردن و بخش Main تفاوتی نکرده است


def plot_multi_domain_history(history, class_colors):
    if len(history) < 3:
        print("تعداد عکس‌های تولید شده برای رسم ۳ سطر کافی نیست!")
        return

    selected_records = random.sample(history, 3)
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    
    for row, record in enumerate(selected_records):
        
        # 1. ستون اول: عکس تارگت اصلی (پس‌زمینه)
        target_vis = draw_yolo_polygons(record['target_img'], record['target_txt'], class_colors)
        axes[row, 0].imshow(target_vis)
        axes[row, 0].set_title(f"1. Target (Background)\n{os.path.basename(record['target_img'])}", fontsize=11)
        axes[row, 0].axis('off')

        # 2. ستون دوم: سورس اول
        if len(record['source_imgs']) > 0:
            src1_vis = draw_yolo_polygons(record['source_imgs'][0], record['source_txts'][0], class_colors)
            axes[row, 1].imshow(src1_vis)
            ds_name = os.path.basename(os.path.dirname(os.path.dirname(record['source_imgs'][0]))) 
            axes[row, 1].set_title(f"2. Source Tool 1 ({ds_name})\n{os.path.basename(record['source_imgs'][0])}", fontsize=11)
        else:
            axes[row, 1].imshow(np.zeros((500, 500, 3), dtype=np.uint8))
            axes[row, 1].set_title("No Source Tool", fontsize=11)
        axes[row, 1].axis('off')

        # 3. ستون سوم: سورس دوم
        if len(record['source_imgs']) > 1:
            src2_vis = draw_yolo_polygons(record['source_imgs'][1], record['source_txts'][1], class_colors)
            axes[row, 2].imshow(src2_vis)
            ds_name = os.path.basename(os.path.dirname(os.path.dirname(record['source_imgs'][1])))
            axes[row, 2].set_title(f"3. Source Tool 2 ({ds_name})\n{os.path.basename(record['source_imgs'][1])}", fontsize=11)
        else:
            blank_img = np.ones((500, 500, 3), dtype=np.uint8) * 50
            axes[row, 2].imshow(blank_img)
            axes[row, 2].set_title("No 2nd Tool Present", fontsize=11, color='gray')
        axes[row, 2].axis('off')

        # 4. ستون چهارم: عکس نهایی
        final_vis = draw_yolo_polygons(record['final_img'], record['final_txt'], class_colors)
        axes[row, 3].imshow(final_vis)
        axes[row, 3].set_title(f"4. Final Composited Result\nAll Tools Injected", fontsize=11, color='darkred', fontweight='bold')
        axes[row, 3].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# بخش اجرایی (Main)
# ==========================================
if __name__ == "__main__":
    SOURCES_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\sources"          
    OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Fake_dataset\augmented_dataset_method33" 

    CORNEA_ID = 3            
    TOOL_IDS = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]  
    
    datasets = ['LMM', 'cadis', 'cataract-1k']
    compositor = YoloClinicallyAwareCompositor(CORNEA_ID, TOOL_IDS, out_dir=OUTPUT_DIR)

    # 1. ساخت استخر ابزارها بدون هیچ‌گونه استثنایی برای کلاس 5
    source_pools = {ds: {t_id: [] for t_id in TOOL_IDS} for ds in datasets}

    for ds in datasets:
        images = glob.glob(os.path.join(SOURCES_DIR, ds, 'images', '*.jpg'))
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
            
            for tool_id in TOOL_IDS:
                if tool_id in class_counts:
                    # شرط مربوط به کلاس 5 به صورت کامل حذف شد
                    source_pools[ds][tool_id].append(img_path)

    # 2. تولید دیتاست و دریافت تاریخچه
    history = generate_yolo_dataset(compositor, datasets, source_pools, TOOL_IDS, SOURCES_DIR)

    # 3. پلات نتایج
    CLASS_COLORS = {
        0: (0, 255, 0),      
        1: (255, 0, 0),      
        2: (0, 0, 255),      
        3: (255, 255, 0),    
        4: (255, 0, 255),
        5: (0, 255, 255),
        6: (128, 0, 128),
        7: (128, 128, 0),
        8: (0, 128, 128),
        10: (255, 165, 0),
        11: (255, 192, 203)
    }
    
    if history:
        plot_multi_domain_history(history, CLASS_COLORS)