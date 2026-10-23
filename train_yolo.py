import os
import sys
import argparse
import torch
from ultralytics import YOLO

def check_environment():
    print("=" * 50)
    print("Environment Verification:")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print("=" * 50)
        return True
    else:
        print("⚠️ Warning: CUDA GPU not detected. Training will fall back to CPU or fail if CUDA required.")
        print("=" * 50)
        return False

def start_training(args):
    print(f"\n🚀 Initializing YOLOv11-seg Training with weights: {args.weights}")
    model = YOLO(args.weights)
    
    model.train(
        # --- Basics ---
        data=args.data,
        epochs=args.epochs,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save=True,
        plots=True,

        # --- Research Integrity & Seed ---
        project=args.project,
        name=args.name,
        seed=42,
        deterministic=True,
        patience=args.patience,

        # --- Class Imbalance & Loss ---
        cls_pw=0.0,
        overlap_mask=True,

        # --- Baseline Augmentation (from Paper) ---
        augment=True,
        fliplr=0.5,
        degrees=0.20,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.5,
        hsv_v=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0
    )
    print("\n✅ Training complete.")

def run_validation(args):
    if not os.path.exists(args.weights):
        print(f"❌ Error: Model weights not found at {args.weights}")
        return

    print(f"\n📊 Starting Validation using model: {args.weights}")
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device
    )
    print("✅ Validation completed successfully!")
    return metrics

def run_prediction(args):
    if not os.path.exists(args.weights):
        print(f"❌ Error: Model weights not found at {args.weights}")
        return
        
    if not os.path.exists(args.source):
        print(f"❌ Error: Source path not found at {args.source}")
        return

    print(f"\n🔍 Running Prediction (Inference)...")
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        device=args.device
    )
    print(f"✅ Prediction completed! Saved to runs/segment/predict")
    return results

def main():
    parser = argparse.ArgumentParser(description="YOLOv11-seg Pipeline for Cataract Surgery Domain Generalization")
    parser.add_argument("--mode", type=str, choices=["train", "val", "predict"], default="train", help="Mode: train, val, predict")
    parser.add_argument("--weights", type=str, default="yolo11m-seg.pt", help="Path to initial weights or trained model")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml config")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size (pixels)")
    parser.add_argument("--device", default=0, help="Device (0 or cpu)")
    parser.add_argument("--lr0", type=float, default=0.000625, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD / Adam momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (AdamW, SGD, auto)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="Cataract_DG", help="Project directory name")
    parser.add_argument("--name", type=str, default="exp_run", help="Run experiment name")
    parser.add_argument("--source", type=str, default="./data/test/images", help="Inference source folder or image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for prediction")

    args = parser.parse_args()

    check_environment()

    if args.mode == "train":
        start_training(args)
    elif args.mode == "val":
        run_validation(args)
    elif args.mode == "predict":
        run_prediction(args)

if __name__ == "__main__":
    main()
