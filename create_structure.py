import os

# اسم پوشه اصلی پروژه
root = "Segmentation paper DG"

# ساختار پوشه‌ها
folders = [
    "data",
    "src",
    "notebooks",
    "configs",
    "outputs"
]

# ساخت پوشه اصلی
os.makedirs(root, exist_ok=True)

# ساخت زیرپوشه‌ها
for folder in folders:
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)

# ساخت فایل‌ها
files = {
    ".gitignore": "# Ignore data files\n/data/\n__pycache__/\n*.pyc\n.env\n",
    "README.md": "# Project Title\n\nDescription of your project.\n",
    "requirements.txt": "# Add your dependencies here\nnumpy\npandas\n"
}

for filename, content in files.items():
    file_path = os.path.join(root, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("Project structure created successfully!")