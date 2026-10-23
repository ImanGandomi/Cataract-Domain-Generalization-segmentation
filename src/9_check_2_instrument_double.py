import os
import yaml
from collections import Counter, defaultdict

# ----------------------------
# Load class names from YAML
# ----------------------------
def load_class_names(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])

    # YOLO YAML can store names as:
    # 1) a list: ["cat", "dog", ...]
    # 2) a dict: {0: "cat", 1: "dog", ...}
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]

    return names


# ----------------------------
# Find images where a class appears at least twice
# ----------------------------
def find_duplicate_classes(labels_dir, class_names, image_exts=(".jpg", ".jpeg", ".png", ".bmp")):
    results = []

    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(labels_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Extract class IDs from the first value in each line
        class_ids = []
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            class_ids.append(int(parts[0]))

        counts = Counter(class_ids)
        duplicate_classes = {cls_id: cnt for cls_id, cnt in counts.items() if cnt >= 2}

        if duplicate_classes:
            image_name_base = os.path.splitext(filename)[0]
            image_name = None

            # Try to find the matching image file
            for ext in image_exts:
                candidate = image_name_base + ext
                if os.path.exists(os.path.join(labels_dir, candidate)):
                    image_name = candidate
                    break

            # If image is not in the same folder, just return the base name
            if image_name is None:
                image_name = image_name_base

            readable_duplicates = {
                class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}": cnt
                for cls_id, cnt in duplicate_classes.items()
            }

            results.append({
                "image_name": image_name,
                "duplicates": readable_duplicates
            })

    return results


# ----------------------------
# Main
# ----------------------------
labels_dir = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Youtub\txt\with_padding_114\labels"   # Folder containing YOLO .txt labels
yaml_path = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract_LMM\new_2\txt\raw\data.yaml"       # Path to your YAML file

class_names = load_class_names(yaml_path)
results = find_duplicate_classes(labels_dir, class_names)

# Print results
for item in results:
    print(f"Image: {item['image_name']}")
    for class_name, count in item["duplicates"].items():
        print(f"  - Class: {class_name} | Count: {count}")
    print("-" * 40)
