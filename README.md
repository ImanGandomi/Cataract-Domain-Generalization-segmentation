# Benchmark for Domain Generalization in Surgical Instrument Instance Segmentation

This project investigates the challenge of **Domain Shift** in cataract surgery instrument segmentation. We evaluate how models trained on specific clinical datasets generalize to unseen data from different sources, including diverse clinical centers and YouTube videos.

## 🚀 Project Overview
In medical imaging, especially robotic surgery, models often fail when encountering data from a new hospital or camera setup. This research follows a three-phase benchmark approach to demonstrate this failure and propose both data-centric and model-centric solutions.

## 📊 Datasets
We utilize four distinct datasets with varying resolutions and sources:
* **CADIS**: 4,670 images (1920x1080)
* **Cataract-1k**: 2,450 images (1000x800)
* **Cataract-LMM**: 6,094 images (740x520)
* **YouTube Dataset**: 1,000 images (Unseen Domain)

## 🔬 Research Methodology

### Phase 1: Proving Domain Shift
Training on a single dataset and testing on the remaining ones to quantify performance drop across different clinical domains.

### Phase 2: Data-Centric Generalization
Combining multiple clinical datasets using **Weighted Sampling** to balance domain representation and evaluating the model on the unseen YouTube dataset.

### Phase 3: Model-Centric Optimization
Implementing architectural improvements and advanced loss functions (e.g., Dice + Weighted BCE) to further enhance cross-domain robustness.

## 🛠 Tech Stack
* **Language:** Python
* **Framework:** PyTorch / Detectron2 (or MMDetection)
* **Image Processing:** OpenCV, Albumentations
* **Versioning:** Git & GitHub

## 📁 Project Structure
```text
├── data/               # Raw and Preprocessed datasets
├── src/                # Source code (Preprocessing, Training, Inference)
├── configs/            # Model configurations and Hyperparameters
├── outputs/            # Saved weights and evaluation logs
├── notebooks/          # Experimental analysis and visualizations
└── requirements.txt    # Project dependencies