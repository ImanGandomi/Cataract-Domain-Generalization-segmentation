import json
import re
from collections import defaultdict
from itertools import combinations

def find_all_unique_partitions(json_path, target_sums):
    # ۱. بارگذاری و استخراج اطلاعات ویدیوها
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    video_counts = defaultdict(int)
    for img in coco_data['images']:
        # اعمال اصلاحیه کاربر: تبدیل case_ به case
        match = re.search(r'case(\d+)', img['file_name'])
        if match:
            vid_id = match.group(0)
            video_counts[vid_id] += 1

    video_items = sorted(list(video_counts.items())) # مرتب‌سازی برای ثبات در محاسبات
    num_vids_per_fold = 6
    
    # ۲. پیش‌محاسبه تمام گروه‌های ۶تایی منحصربه‌فرد برای هر عدد
    print("⏳ Calculating unique 6-video groups for each target...")
    combos_per_target = {}
    for target in target_sums:
        # استفاده از set برای اطمینان از اینکه جابجایی ویدیوها حالت جدید ایجاد نمی‌کند
        valid_groups = []
        for combo in combinations(video_items, num_vids_per_fold):
            if sum(item[1] for item in combo) == target:
                # ذخیره به صورت frozenset برای قابلیت هش شدن و مقایسه بدون ترتیب
                valid_groups.append(frozenset(item[0] for item in combo))
        combos_per_target[target] = valid_groups
        print(f"   - Target {target}: Found {len(valid_groups)} unique groups.")

    all_solutions = set() # استفاده از set برای حذف تکرار احتمالی کل راه‌حل

    # ۳. تابع جستجوی بازگشتی (Backtracking)
    def solve(target_idx, used_vids, current_solution):
        if target_idx == len(target_sums):
            # تبدیل راه‌حل به یک tuple مرتب شده از فریز‌ن‌ست‌ها برای اضافه شدن به set نهایی
            # این کار باعث می‌شود جابجایی فولدها هم حالت تکراری ایجاد نکند
            all_solutions.add(tuple(sorted(current_solution, key=lambda x: sorted(list(x[1])))))
            return

        target = target_sums[target_idx]
        for group in combos_per_target[target]:
            if not (group & used_vids): # اگر تداخلی با ویدیوهای قبلی نداشت
                current_solution.append((target, group))
                solve(target_idx + 1, used_vids | group, current_solution)
                current_solution.pop()

    # ۴. اجرا و نمایش
    print("🔍 Searching for all valid disjoint partitions...")
    solve(0, set(), [])

    if not all_solutions:
        print("\n❌ هیچ ترکیب ۵-فولدی با این مشخصات پیدا نشد.")
    else:
        print(f"\n✨ Found {len(all_solutions)} unique partitions!")
        print("="*60)
        for i, sol in enumerate(all_solutions):
            print(f"🌟 Solution #{i+1}:")
            for target, vids in sorted(list(sol)):
                # نمایش ویدیوها به صورت مرتب شده برای خوانایی بیشتر
                print(f"   Target {target}: {sorted(list(vids))}")
            print("-" * 40)

# تنظیمات
TARGETS = [561, 459, 420, 385, 431]
JSON_PATH = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\raw\_annotations.coco.json"

if __name__ == "__main__":
    find_all_unique_partitions(JSON_PATH, TARGETS)