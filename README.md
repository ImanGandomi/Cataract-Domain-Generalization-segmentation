# Decoupling Context from Morphology: A Data-Centric Cross-Domain Compositing Framework for Cataract Surgery Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11-blue.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)

Official repository for **"Decoupling Context from Morphology: A Data-Centric Cross-Domain Compositing Framework for Cataract Surgery Segmentation"**.

---

## 📌 Abstract

The clinical translation of surgical artificial intelligence is severely constrained by **domain shift**, where models trained on specific institutional data degrade significantly when deployed in novel operating environments. In this work, we quantify this degradation in cataract surgery instrument segmentation across three clinical datasets (**CaDIS**, **Cataract-1K**, **Cataract-LMM**) and a newly curated, unconstrained out-of-distribution (OOD) test cohort sourced from global YouTube videos.

We demonstrate that standard segmentation networks overfit to spurious contextual correlations, associating surgical tools with institutional background lighting, microscope optics, and tissue textures rather than learning intrinsic tool morphology. 

To overcome this limitation, we introduce **Cross-Domain Hybrid Compositing (CDHC)**, an architecture-agnostic, data-centric framework that achieves **contextual decoupling**. By transplanting isolated instruments onto foreign background frames using anatomy-aware scaling and adaptive illumination matching, our approach forces networks to rely on domain-invariant morphological features. Applied across two-stage, single-stage, and vision transformer architectures without structural modifications or inference latency, this approach elevates zero-shot OOD accuracy on the unconstrained YouTube cohort from a baseline of **57.4% to 74.4% (Mask mAP50)**.

---

## 🔬 Benchmark Overview

Our study is organized into three progressive experimental phases:

```
[Phase 1: Proving Domain Shift]
Single-Center Training (CaDIS / Cataract-1K / Cataract-LMM) ───► Severe performance drop on cross-center splits

[Phase 2: Multi-Source Joint Training]
Naive Multi-Center Merging ───────────────────────────────────► Modest improvement, plateaus around ~70% OOD mAP50

[Phase 3: Contextual Decoupling (CDHC)]
Cross-Domain Hybrid Compositing Augmentation ──────────────────► Decouples tool geometry from native background,
                                                                 achieving 74.4% mAP50 on in-the-wild YouTube cohort
```

### Dataset Characteristics & Unified Taxonomy

To resolve taxonomy discrepancies across independent datasets, all annotations are mapped into a standardized **12-class schema**:
`['Cannula', 'Cap Cystotome', 'Cap Forceps', 'Cornea', 'Forceps', 'I-A Handpiece', 'Lens Injector', 'Phaco Handpiece', 'Primary Knife', 'Pupil', 'Second Instrument', 'Secondary Knife']`

| Dataset | Type | Frames | Resolution | Domain Characteristics |
| :--- | :---: | :---: | :---: | :---: |
| **CaDIS** | Single-center | 4,670 | 1920 × 1080 | Uniform illumination, high resolution |
| **Cataract-1K** | Single-center | 2,256 | 1000 × 800 | Stable microscope optics |
| **Cataract-LMM** | Multi-center | 6,094 | 740 × 520 | Heterogeneous clinical settings |
| **YouTube Cohort** | In-the-Wild (OOD) | 921 | Diverse | Extreme lighting shifts, motion blur, diverse patients |

---

## 📊 Experimental Results

### Zero-Shot Generalization on the "In-The-Wild" YouTube Cohort:

| Experimental Phase | Training Setup | Architecture | In-Domain Test (mAP50) | YouTube OOD (mAP50) |
| :--- | :--- | :---: | :---: | :---: |
| **Phase 1** | Single-Source (Cataract-LMM) | YOLOv11m-seg | 84.1% | 57.4% |
| **Phase 2** | Multi-Source Joint Training | YOLOv11m-seg | 88.5% | 70.1% |
| **Phase 3 (Ours)** | **Contextual Decoupling (CDHC)** | **YOLOv11m-seg** | **90.2%** | **74.4% (+17.0%)** |
| **Phase 3 (Ours)** | **Contextual Decoupling (CDHC)** | **Mask R-CNN** | **87.8%** | **71.2% (+13.8%)** |

---

## 📁 Repository Structure

```text
├── configs/
│   └── data.yaml                   # YOLO dataset configuration & class names
├── src/
│   ├── 1_fix_case_names.py         # Harmonize file naming conventions
│   ├── 2_preprocess.py             # Preprocessing & image padding
│   ├── 3_visualize_check_pad.py    # Verification of padding and aspect ratios
│   ├── 4_split_data.py             # Video-level train/val/test splitting
│   ├── 5_coco_to_yolo.py           # Format conversion from COCO JSON to YOLO TXT
│   ├── 6_visualize_check_coco2txt.py# Sanity check for bounding box and polygon conversions
│   ├── 7_check_txt_label.py        # Label verification
│   ├── create_fake_dataset.py      # Core Cross-Domain Hybrid Compositing engine
│   ├── create_fake_dataset_method2.py # Variant with advanced illumination blending
│   ├── calculate_final_metrics_yolo.py# Evaluation script for Mask mAP and metrics
│   ├── dataset.py                  # PyTorch Dataset implementation for Mask R-CNN
│   ├── model.py                    # Mask R-CNN backbone definitions
│   └── transforms.py               # Custom transforms and augmentations
├── train_yolo.py                   # Main CLI for YOLOv11-seg (train, val, predict)
├── train_mask_rcnn.py              # Training pipeline for Mask R-CNN
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md
```

---

## 🛠️ Getting Started

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/ImanGandomi/Cataract-Domain-Generalization-segmentation.git
cd Cataract-Domain-Generalization-segmentation

# Create and activate environment
conda create -n cataract-dg python=3.10 -y
conda activate cataract-dg

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation & Preprocessing

1. Place raw datasets in the `data/` directory according to [data/README.md](data/README.md).
2. Execute the preprocessing pipeline:
```bash
python src/2_preprocess.py
python src/4_split_data.py
python src/5_coco_to_yolo.py
```

### 3. Generate Cross-Domain Composited Dataset (CDHC)

Run the contextual decoupling generator to transplant isolated surgical instruments onto instrument-free background frames from other domains:

```bash
python src/create_fake_dataset.py
```

---

## 🏋️ Training & Evaluation

### Training YOLOv11m-seg
```bash
# Train on the composited dataset using AdamW optimizer
python train_yolo.py --mode train --data configs/data.yaml --epochs 100 --batch 16 --imgsz 512 --weights yolo11m-seg.pt
```

### Evaluating on the Out-of-Distribution Cohort
```bash
python train_yolo.py --mode val --data configs/data.yaml --weights runs/segment/Cataract_DG/exp_run/weights/best.pt
```

### Running Inference on New Surgical Videos/Images
```bash
python train_yolo.py --mode predict --source path/to/unseen_frames/ --weights path/to/best.pt --conf 0.25
```

---

## 📑 Citation

If you find this research, dataset benchmark, or compositing framework useful in your work, please cite:

```bibtex
@article{gandomi2026decoupling,
  title={Decoupling Context from Morphology: A Data-Centric Cross-Domain Compositing Framework for Cataract Surgery Segmentation},
  author={Gandomi, Iman},
  journal={arXiv preprint},
  year={2026}
}
```

---

## ✉️ Contact & Acknowledgements

* **Author:** Iman Gandomi
* **Affiliation:** [Applied Robotics and AI Solutions (ARAS)](https://aras.kntu.ac.ir/), Faculty of Electrical Engineering, K.N. Toosi University of Technology, Tehran, Iran.
* For research collaborations and inquiries, please reach out via GitHub issues or email.
