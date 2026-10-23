# import re
# import random
# from pathlib import Path
# from collections import defaultdict, Counter

# import numpy as np
# import pandas as pd


# # ==========================================================
# # CONFIG
# # ==========================================================

# IMAGE_DIR = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\raw\train\images")
# LABEL_DIR = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\raw\train\labels")

# TRAIN_RATIO = 0.70
# VALID_RATIO = 0.15
# TEST_RATIO = 0.15

# RANDOM_SEED = 42
# N_ITER = 5000

# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# # ==========================================================
# # PARSE FILE NAME
# # ==========================================================

# PATTERN = re.compile(
#     r"SE_(\d+)_(\d+)_(S\d+)"
# )


# def parse_filename(name):
#     m = PATTERN.search(name)

#     if m is None:
#         return None

#     video_id = m.group(1)
#     frame_id = m.group(2)
#     center = m.group(3)

#     return video_id, frame_id, center


# # ==========================================================
# # ANALYZE DATASET
# # ==========================================================

# video_info = {}

# global_class_counter = Counter()

# all_images = list(IMAGE_DIR.glob("*"))

# for img_path in all_images:

#     parsed = parse_filename(img_path.stem)

#     if parsed is None:
#         continue

#     video_id, frame_id, center = parsed

#     label_path = LABEL_DIR / f"{img_path.stem}.txt"

#     if video_id not in video_info:

#         video_info[video_id] = {
#             "center": center,
#             "frames": 0,
#             "class_counts": Counter(),
#         }

#     video_info[video_id]["frames"] += 1

#     if label_path.exists():

#         with open(label_path, "r") as f:

#             for line in f:

#                 line = line.strip()

#                 if len(line) == 0:
#                     continue

#                 cls = int(line.split()[0])

#                 video_info[video_id]["class_counts"][cls] += 1
#                 global_class_counter[cls] += 1


# # ==========================================================
# # GLOBAL STATISTICS
# # ==========================================================

# total_frames = sum(v["frames"] for v in video_info.values())

# videos = list(video_info.keys())

# centers = Counter(
#     v["center"]
#     for v in video_info.values()
# )

# center_frames = Counter()

# for v in video_info.values():
#     center_frames[v["center"]] += v["frames"]

# print("\n==========================")
# print("GLOBAL DATASET STATISTICS")
# print("==========================")

# print("Total images :", total_frames)
# print("Total videos :", len(video_info))

# print("\nVideos per center")
# for c, n in centers.items():
#     print(c, n)

# print("\nFrames per center")
# for c, n in center_frames.items():
#     print(c, n)

# print("\nClass distribution")

# for cls in sorted(global_class_counter):

#     cnt = global_class_counter[cls]

#     pct = 100 * cnt / sum(global_class_counter.values())

#     print(
#         f"Class {cls}: "
#         f"{cnt} instances "
#         f"({pct:.2f}%)"
#     )

# # ==========================================================
# # VIDEO TABLE
# # ==========================================================

# video_rows = []

# all_classes = sorted(global_class_counter.keys())

# for vid, info in video_info.items():

#     row = {
#         "video_id": vid,
#         "center": info["center"],
#         "frames": info["frames"]
#     }

#     for cls in all_classes:
#         row[f"class_{cls}"] = info["class_counts"][cls]

#     video_rows.append(row)

# video_df = pd.DataFrame(video_rows)

# # ==========================================================
# # COST FUNCTION
# # ==========================================================

# target_ratios = {
#     "train": TRAIN_RATIO,
#     "valid": VALID_RATIO,
#     "test": TEST_RATIO
# }


# def compute_stats(split_videos):

#     frames = 0

#     center_counter = Counter()

#     class_counter = Counter()

#     for vid in split_videos:

#         info = video_info[vid]

#         frames += info["frames"]

#         center_counter[info["center"]] += info["frames"]

#         class_counter.update(
#             info["class_counts"]
#         )

#     return frames, center_counter, class_counter


# def score_split(train, valid, test):

#     splits = {
#         "train": train,
#         "valid": valid,
#         "test": test
#     }

#     score = 0.0

#     # ----------------------------------
#     # frame ratio
#     # ----------------------------------

#     for name, vids in splits.items():

#         frames, _, _ = compute_stats(vids)

#         actual_ratio = frames / total_frames

#         target = target_ratios[name]

#         score += 50 * abs(
#             actual_ratio - target
#         )

#     # ----------------------------------
#     # class distribution
#     # ----------------------------------

#     global_total = sum(
#         global_class_counter.values()
#     )

#     global_pct = {}

#     for cls, cnt in global_class_counter.items():
#         global_pct[cls] = cnt / global_total

#     for name, vids in splits.items():

#         _, _, cls_counter = compute_stats(
#             vids
#         )

#         split_total = sum(
#             cls_counter.values()
#         )

#         if split_total == 0:
#             score += 1000
#             continue

#         for cls in global_pct:

#             split_pct = (
#                 cls_counter[cls]
#                 / split_total
#             )

#             score += 200 * abs(
#                 split_pct
#                 - global_pct[cls]
#             )

#     # ----------------------------------
#     # center distribution
#     # ----------------------------------

#     global_center_total = sum(
#         center_frames.values()
#     )

#     global_center_pct = {
#         k: v / global_center_total
#         for k, v in center_frames.items()
#     }

#     for name, vids in splits.items():

#         _, center_cnt, _ = compute_stats(
#             vids
#         )

#         total_split_frames = sum(
#             center_cnt.values()
#         )

#         if total_split_frames == 0:
#             score += 1000
#             continue

#         for center in global_center_pct:

#             if center_cnt[center] == 0:
#                 score += 500

#             split_pct = (
#                 center_cnt[center]
#                 / total_split_frames
#             )

#             score += 100 * abs(
#                 split_pct
#                 - global_center_pct[center]
#             )

#     return score


# # ==========================================================
# # RANDOM SEARCH OPTIMIZATION
# # ==========================================================

# video_ids = list(video_info.keys())

# best_score = np.inf
# best_split = None

# for _ in range(N_ITER):

#     shuffled = video_ids.copy()
#     random.shuffle(shuffled)

#     train = []
#     valid = []
#     test = []

#     frame_train = 0
#     frame_valid = 0
#     frame_test = 0

#     for vid in shuffled:

#         n_frames = video_info[vid]["frames"]

#         current = {
#             "train":
#                 frame_train / total_frames,
#             "valid":
#                 frame_valid / total_frames,
#             "test":
#                 frame_test / total_frames
#         }

#         deficits = {
#             "train":
#                 TRAIN_RATIO - current["train"],
#             "valid":
#                 VALID_RATIO - current["valid"],
#             "test":
#                 TEST_RATIO - current["test"]
#         }

#         target_split = max(
#             deficits,
#             key=deficits.get
#         )

#         if target_split == "train":
#             train.append(vid)
#             frame_train += n_frames

#         elif target_split == "valid":
#             valid.append(vid)
#             frame_valid += n_frames

#         else:
#             test.append(vid)
#             frame_test += n_frames

#     s = score_split(
#         train,
#         valid,
#         test
#     )

#     if s < best_score:

#         best_score = s

#         best_split = (
#             train,
#             valid,
#             test
#         )

# # ==========================================================
# # FINAL REPORT
# # ==========================================================

# train, valid, test = best_split


# def summarize(name, vids):

#     frames, centers_cnt, cls_cnt = \
#         compute_stats(vids)

#     return {
#         "Split": name,
#         "Videos": len(vids),
#         "Frames": frames,
#         "S1 Frames":
#             centers_cnt["S1"],
#         "S2 Frames":
#             centers_cnt["S2"]
#     }


# summary_df = pd.DataFrame([
#     summarize("Train", train),
#     summarize("Valid", valid),
#     summarize("Test", test)
# ])

# print("\n")
# print(summary_df)

# # ==========================================================
# # CLASS DISTRIBUTION REPORT
# # ==========================================================

# rows = []

# for cls in all_classes:

#     total_pct = (
#         global_class_counter[cls]
#         / sum(global_class_counter.values())
#     ) * 100

#     split_stats = []

#     for vids in [train, valid, test]:

#         _, _, cc = compute_stats(vids)

#         total_split = sum(cc.values())

#         if total_split == 0:
#             split_stats.append(0)
#         else:
#             split_stats.append(
#                 100 * cc[cls] / total_split
#             )

#     rows.append([
#         cls,
#         total_pct,
#         *split_stats
#     ])

# class_report = pd.DataFrame(
#     rows,
#     columns=[
#         "Class",
#         "Total %",
#         "Train %",
#         "Valid %",
#         "Test %"
#     ]
# )

# # ==========================================================
# # SAVE VIDEO LISTS
# # ==========================================================

# with open(
#         "train_videos.txt",
#         "w") as f:

#     for v in sorted(train):
#         f.write(f"{v}\n")

# with open(
#         "valid_videos.txt",
#         "w") as f:

#     for v in sorted(valid):
#         f.write(f"{v}\n")

# with open(
#         "test_videos.txt",
#         "w") as f:

#     for v in sorted(test):
#         f.write(f"{v}\n")

# # ==========================================================
# # EXCEL REPORT
# # ==========================================================

# with pd.ExcelWriter(
#         "split_report.xlsx"
# ) as writer:

#     summary_df.to_excel(
#         writer,
#         sheet_name="Summary",
#         index=False
#     )

#     class_report.to_excel(
#         writer,
#         sheet_name="ClassDistribution",
#         index=False
#     )

#     video_df.to_excel(
#         writer,
#         sheet_name="VideoStatistics",
#         index=False
#     )

# print("\nBest score:", best_score)
# print("Report saved -> split_report.xlsx")


###################################################################################################

# import re
# from pathlib import Path
# from collections import Counter, defaultdict

# import pandas as pd


# # ==========================================================
# # CONFIG
# # ==========================================================

# DATASET_ROOT = Path(r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\padded_splited_data")

# SPLITS = {
#     "train": DATASET_ROOT / "train",
#     "valid": DATASET_ROOT / "val",
#     "test": DATASET_ROOT / "test",
# }

# PATTERN = re.compile(
#     r"SE_(\d+)_(\d+)_(S\d+)"
# )


# # ==========================================================
# # PARSER
# # ==========================================================

# def parse_filename(stem):

#     m = PATTERN.search(stem)

#     if m is None:
#         return None

#     return {
#         "video_id": m.group(1),
#         "frame_id": m.group(2),
#         "center": m.group(3)
#     }


# # ==========================================================
# # ANALYZE SPLIT
# # ==========================================================

# all_class_counter = Counter()

# split_stats = {}
# video_table = []

# for split_name, split_dir in SPLITS.items():

#     image_dir = split_dir / "images"
#     label_dir = split_dir / "labels"

#     video_info = {}

#     for img_path in image_dir.glob("*"):

#         parsed = parse_filename(img_path.stem)

#         if parsed is None:
#             continue

#         video_id = parsed["video_id"]
#         center = parsed["center"]

#         if video_id not in video_info:

#             video_info[video_id] = {
#                 "center": center,
#                 "frames": 0,
#                 "class_counts": Counter()
#             }

#         video_info[video_id]["frames"] += 1

#         label_path = label_dir / f"{img_path.stem}.txt"

#         if label_path.exists():

#             with open(label_path, "r") as f:

#                 for line in f:

#                     line = line.strip()

#                     if not line:
#                         continue

#                     cls = int(line.split()[0])

#                     video_info[video_id]["class_counts"][cls] += 1
#                     all_class_counter[cls] += 1

#     split_stats[split_name] = video_info


# # ==========================================================
# # PATIENT-WISE CHECK
# # ==========================================================

# video_to_split = {}

# duplicates = []

# for split_name, vids in split_stats.items():

#     for vid in vids:

#         if vid in video_to_split:

#             duplicates.append(
#                 (vid,
#                  video_to_split[vid],
#                  split_name)
#             )

#         video_to_split[vid] = split_name


# # ==========================================================
# # SUMMARY TABLE
# # ==========================================================

# summary_rows = []

# for split_name, vids in split_stats.items():

#     frames = 0

#     s1_frames = 0
#     s2_frames = 0

#     for vid, info in vids.items():

#         frames += info["frames"]

#         if info["center"] == "S1":
#             s1_frames += info["frames"]

#         if info["center"] == "S2":
#             s2_frames += info["frames"]

#     summary_rows.append({

#         "Split": split_name,

#         "Videos": len(vids),

#         "Frames": frames,

#         "S1 Frames": s1_frames,

#         "S2 Frames": s2_frames
#     })

# summary_df = pd.DataFrame(summary_rows)


# # ==========================================================
# # CLASS DISTRIBUTION
# # ==========================================================

# total_instances = sum(all_class_counter.values())

# class_rows = []

# all_classes = sorted(all_class_counter.keys())

# for cls in all_classes:

#     total_pct = (
#         all_class_counter[cls]
#         / total_instances
#     ) * 100

#     row = {
#         "Class": cls,
#         "Total %": total_pct
#     }

#     for split_name, vids in split_stats.items():

#         split_counter = Counter()

#         for v in vids.values():
#             split_counter.update(
#                 v["class_counts"]
#             )

#         split_total = sum(
#             split_counter.values()
#         )

#         if split_total == 0:
#             pct = 0

#         else:
#             pct = (
#                 split_counter[cls]
#                 / split_total
#             ) * 100

#         row[f"{split_name}_%"] = pct

#     class_rows.append(row)

# class_df = pd.DataFrame(class_rows)


# # ==========================================================
# # VIDEO TABLE
# # ==========================================================

# for split_name, vids in split_stats.items():

#     for vid, info in vids.items():

#         row = {

#             "Split": split_name,

#             "Video ID": vid,

#             "Center": info["center"],

#             "Frames": info["frames"]
#         }

#         for cls in all_classes:

#             row[f"class_{cls}"] = \
#                 info["class_counts"][cls]

#         video_table.append(row)

# video_df = pd.DataFrame(video_table)


# # ==========================================================
# # DISTANCE TO GLOBAL DISTRIBUTION
# # ==========================================================

# global_dist = {}

# for cls in all_classes:

#     global_dist[cls] = (
#         all_class_counter[cls]
#         / total_instances
#     )

# distance_rows = []

# for split_name, vids in split_stats.items():

#     split_counter = Counter()

#     for v in vids.values():

#         split_counter.update(
#             v["class_counts"]
#         )

#     split_total = sum(
#         split_counter.values()
#     )

#     l1_distance = 0

#     for cls in all_classes:

#         split_pct = (
#             split_counter[cls]
#             / split_total
#         ) if split_total > 0 else 0

#         l1_distance += abs(
#             split_pct
#             - global_dist[cls]
#         )

#     distance_rows.append({

#         "Split": split_name,

#         "Distribution_Error":
#             l1_distance
#     })

# distance_df = pd.DataFrame(distance_rows)


# # ==========================================================
# # SAVE REPORT
# # ==========================================================

# with pd.ExcelWriter(
#         "existing_split_report.xlsx"
# ) as writer:

#     summary_df.to_excel(
#         writer,
#         sheet_name="Summary",
#         index=False
#     )

#     class_df.to_excel(
#         writer,
#         sheet_name="ClassDistribution",
#         index=False
#     )

#     video_df.to_excel(
#         writer,
#         sheet_name="VideoStatistics",
#         index=False
#     )

#     distance_df.to_excel(
#         writer,
#         sheet_name="DistributionError",
#         index=False
#     )


# # ==========================================================
# # PRINT
# # ==========================================================

# print("\n========================")
# print("SUMMARY")
# print("========================")
# print(summary_df)

# print("\n")

# print(class_df)

# print("\n")

# print(distance_df)

# if len(duplicates) == 0:

#     print(
#         "\nPatient-wise check: PASSED"
#     )

# else:

#     print(
#         "\nPatient-wise check: FAILED"
#     )

#     for d in duplicates:
#         print(d)

# print(
#     "\nReport saved: "
#     "existing_split_report.xlsx"
# )



############################################################ for 2 split comparison
# import pandas as pd
# import numpy as np


# # ==========================================================
# # CONFIG
# # ==========================================================

# CURRENT_REPORT = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\split2\existing_split_report.xlsx"
# PROPOSED_REPORT = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\split1\split_report.xlsx"

# TARGET_RATIOS = {
#     "train": 0.70,
#     "valid": 0.15,
#     "test": 0.15
# }


# # ==========================================================
# # LOAD
# # ==========================================================

# def load_report(path):

#     summary = pd.read_excel(
#         path,
#         sheet_name="Summary"
#     )

#     classes = pd.read_excel(
#         path,
#         sheet_name="ClassDistribution"
#     )

#     return summary, classes


# # ==========================================================
# # CLASS ERROR
# # ==========================================================

# def class_distribution_error(class_df):

#     total_col = "Total %"

#     split_cols = [
#         c for c in class_df.columns
#         if c.endswith("%")
#         and c != total_col
#     ]

#     errors = {}

#     for col in split_cols:

#         err = np.abs(
#             class_df[col]
#             - class_df[total_col]
#         ).sum()

#         errors[col] = err

#     return errors


# # ==========================================================
# # FRAME RATIO ERROR
# # ==========================================================

# def frame_ratio_error(summary_df):

#     total_frames = summary_df["Frames"].sum()

#     errors = {}

#     for _, row in summary_df.iterrows():

#         split = row["Split"].lower()

#         actual_ratio = (
#             row["Frames"]
#             / total_frames
#         )

#         target = TARGET_RATIOS[split]

#         errors[split] = abs(
#             actual_ratio
#             - target
#         )

#     return errors


# # ==========================================================
# # CENTER ERROR
# # ==========================================================

# def center_distribution_error(summary_df):

#     global_s1 = (
#         summary_df["S1 Frames"].sum()
#     )

#     global_s2 = (
#         summary_df["S2 Frames"].sum()
#     )

#     global_total = global_s1 + global_s2

#     global_s1_pct = (
#         global_s1 / global_total
#     )

#     global_s2_pct = (
#         global_s2 / global_total
#     )

#     errors = {}

#     for _, row in summary_df.iterrows():

#         split = row["Split"]

#         total = (
#             row["S1 Frames"]
#             + row["S2 Frames"]
#         )

#         if total == 0:

#             errors[split] = 999
#             continue

#         s1_pct = (
#             row["S1 Frames"]
#             / total
#         )

#         s2_pct = (
#             row["S2 Frames"]
#             / total
#         )

#         err = abs(
#             s1_pct - global_s1_pct
#         )

#         err += abs(
#             s2_pct - global_s2_pct
#         )

#         errors[split] = err

#     return errors


# # ==========================================================
# # MISSING CENTER PENALTY
# # ==========================================================

# def missing_center_penalty(summary_df):

#     penalty = 0

#     for _, row in summary_df.iterrows():

#         if row["S1 Frames"] == 0:
#             penalty += 1

#         if row["S2 Frames"] == 0:
#             penalty += 1

#     return penalty


# # ==========================================================
# # EVALUATE
# # ==========================================================

# def evaluate(report_path):

#     summary, classes = load_report(
#         report_path
#     )

#     class_err = class_distribution_error(
#         classes
#     )

#     frame_err = frame_ratio_error(
#         summary
#     )

#     center_err = center_distribution_error(
#         summary
#     )

#     penalty = missing_center_penalty(
#         summary
#     )

#     total_class_err = sum(
#         class_err.values()
#     )

#     total_frame_err = sum(
#         frame_err.values()
#     )

#     total_center_err = sum(
#         center_err.values()
#     )

#     final_score = (

#         4 * total_class_err

#         + 3 * total_center_err

#         + 1 * total_frame_err

#         + 100 * penalty
#     )

#     return {

#         "ClassError":
#             total_class_err,

#         "CenterError":
#             total_center_err,

#         "RatioError":
#             total_frame_err,

#         "MissingCenterPenalty":
#             penalty,

#         "FinalScore":
#             final_score
#     }


# # ==========================================================
# # RUN
# # ==========================================================

# current = evaluate(
#     CURRENT_REPORT
# )

# proposed = evaluate(
#     PROPOSED_REPORT
# )

# rows = []

# for metric in current.keys():

#     cur = current[metric]
#     prop = proposed[metric]

#     better = (
#         "Current"
#         if cur < prop
#         else "Proposed"
#     )

#     rows.append([

#         metric,

#         cur,

#         prop,

#         better
#     ])

# comparison_df = pd.DataFrame(
#     rows,
#     columns=[
#         "Metric",
#         "Current Split",
#         "Proposed Split",
#         "Better"
#     ]
# )

# comparison_df.to_excel(
#     "split_comparison.xlsx",
#     index=False
# )

# print("\n")
# print(comparison_df)

# print("\n")

# current_score = current["FinalScore"]
# proposed_score = proposed["FinalScore"]

# if proposed_score < current_score:

#     print(
#         "WINNER = Proposed Split"
#     )

# else:

#     print(
#         "WINNER = Current Split"
#     )

# print(
#     "\nSaved: split_comparison.xlsx"
# )



######################################################################## for propose change in my current split

# import pandas as 




# import pandas as pd

# # ======================================================
# # CONFIG
# # ======================================================

# REPORT = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\split2\existing_split_report.xlsx"

# SCENARIOS = {

#     "Move_81": [81],

#     "Move_41": [41],

#     "Move_81_41": [81, 41],
# }

# # ======================================================
# # LOAD
# # ======================================================

# df = pd.read_excel(
#     REPORT,
#     sheet_name="VideoStatistics"
# )

# df = df.dropna(
#     subset=["Video ID"]
# )

# df["Split"] = (
#     df["Split"]
#     .astype(str)
#     .str.lower()
# )

# df["Center"] = (
#     df["Center"]
#     .astype(str)
# )

# df["Video ID"] = (
#     df["Video ID"]
#     .astype(int)
# )

# # ======================================================
# # CLASS COLUMNS
# # ======================================================

# class_cols = [

#     c

#     for c in df.columns

#     if str(c).startswith("class_")
# ]

# print("Classes:")
# print(class_cols)

# # ======================================================
# # ANALYZE
# # ======================================================

# def analyze(dataset_df):

#     rows = []

#     for cls_col in class_cols:

#         cls_id = cls_col.replace(
#             "class_",
#             ""
#         )

#         row = {

#             "Class": cls_id,

#             "Total":
#                 dataset_df[cls_col].sum()
#         }

#         for split in [
#             "train",
#             "valid",
#             "test"
#         ]:

#             split_df = dataset_df[
#                 dataset_df["Split"] == split
#             ]

#             row[
#                 f"{split}_total"
#             ] = split_df[
#                 cls_col
#             ].sum()

#             for center in [
#                 "S1",
#                 "S2"
#             ]:

#                 value = split_df.loc[
#                     split_df["Center"]
#                     == center,
#                     cls_col
#                 ].sum()

#                 row[
#                     f"{split}_{center}"
#                 ] = value

#         rows.append(row)

#     return pd.DataFrame(rows)


# # ======================================================
# # RUN SCENARIOS
# # ======================================================

# writer = pd.ExcelWriter(
#     "scenario_analysis.xlsx"
# )

# # ----------------------------------
# # ORIGINAL
# # ----------------------------------

# orig = analyze(df)

# orig.to_excel(
#     writer,
#     sheet_name="Original",
#     index=False
# )

# # ----------------------------------
# # SCENARIOS
# # ----------------------------------

# for scenario_name, videos in \
#         SCENARIOS.items():

#     scenario_df = df.copy()

#     mask = (

#         scenario_df["Video ID"]
#         .isin(videos)

#         &

#         scenario_df["Split"]
#         .isin(["valid"])
#     )

#     scenario_df.loc[
#         mask,
#         "Split"
#     ] = "train"

#     result = analyze(
#         scenario_df
#     )

#     result.to_excel(
#         writer,
#         sheet_name=scenario_name,
#         index=False
#     )

#     print(
#         f"Done: {scenario_name}"
#     )

# writer.close()

# print(
#     "\nSaved: scenario_analysis.xlsx"
# )




import pandas as pd

# =====================================================
# CONFIG
# =====================================================

REPORT = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\split2\existing_split_report.xlsx"

VIDEO_A = 80
VIDEO_B = 106

# =====================================================
# LOAD
# =====================================================

df = pd.read_excel(
    REPORT,
    sheet_name="VideoStatistics"
)

df = df.dropna(
    subset=["Video ID"]
)

df["Split"] = (
    df["Split"]
    .astype(str)
    .str.lower()
)

df["Video ID"] = (
    df["Video ID"]
    .astype(int)
)

# =====================================================
# CLASS COLUMNS
# =====================================================

class_cols = [
    c for c in df.columns
    if str(c).startswith("class_")
]

# =====================================================
# TEST SUMMARY
# =====================================================

def summarize_test(dataframe):

    test_df = dataframe[
        dataframe["Split"] == "test"
    ]

    rows = []

    for cls in class_cols:

        total = test_df[cls].sum()

        s1_df = test_df[
            test_df["Center"] == "S1"
        ]

        s2_df = test_df[
            test_df["Center"] == "S2"
        ]

        s1_total = s1_df[cls].sum()
        s2_total = s2_df[cls].sum()

        s1_videos = (
            s1_df.loc[
                s1_df[cls] > 0,
                "Video ID"
            ]
            .nunique()
        )

        s2_videos = (
            s2_df.loc[
                s2_df[cls] > 0,
                "Video ID"
            ]
            .nunique()
        )

        total_videos = (
            test_df.loc[
                test_df[cls] > 0,
                "Video ID"
            ]
            .nunique()
        )

        rows.append({

            "Class": cls,

            "Test Total":
                total,

            "Test S1":
                s1_total,

            "Test S2":
                s2_total,

            "Videos Total":
                total_videos,

            "Videos S1":
                s1_videos,

            "Videos S2":
                s2_videos
        })

    return pd.DataFrame(rows)

# =====================================================
# ORIGINAL
# =====================================================

original = summarize_test(df)

# =====================================================
# MOVE VIDEO 78
# =====================================================

df78 = df.copy()

mask78 = (

    (df78["Split"] == "test")

    &

    (df78["Video ID"] == VIDEO_A)
)

df78.loc[
    mask78,
    "Split"
] = "train"

move78 = summarize_test(df78)

# =====================================================
# MOVE VIDEO 106
# =====================================================

df106 = df.copy()

mask106 = (

    (df106["Split"] == "test")

    &

    (df106["Video ID"] == VIDEO_B)
)

df106.loc[
    mask106,
    "Split"
] = "train"

move106 = summarize_test(df106)

# =====================================================
# DIFFERENCE TABLES
# =====================================================

diff78 = original.copy()

for col in [
    "Test Total",
    "Test S1",
    "Test S2"
]:
    diff78[col] = (
        move78[col]
        - original[col]
    )

diff106 = original.copy()

for col in [
    "Test Total",
    "Test S1",
    "Test S2"
]:
    diff106[col] = (
        move106[col]
        - original[col]
    )

# =====================================================
# SAVE
# =====================================================

with pd.ExcelWriter(
    "test_video_comparison.xlsx"
) as writer:

    original.to_excel(
        writer,
        sheet_name="Original_Test",
        index=False
    )

    move78.to_excel(
        writer,
        sheet_name="Move_80",
        index=False
    )

    diff78.to_excel(
        writer,
        sheet_name="Diff_80",
        index=False
    )

    move106.to_excel(
        writer,
        sheet_name="Move_106",
        index=False
    )

    diff106.to_excel(
        writer,
        sheet_name="Diff_106",
        index=False
    )

print(
    "Saved -> test_video_comparison.xlsx"
)