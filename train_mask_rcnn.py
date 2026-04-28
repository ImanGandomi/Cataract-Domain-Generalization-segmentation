import torch
# import transforms as T # باید یک فایل کمکی برای تبدیل‌ها داشته باشید
from src.dataset import SurgicalDataset
from src.model import get_model_instance_segmentation
from src.transforms import get_transform
import torch
import os

# ۱. تعریف تابع به صورت مستقل (بیرون از main)
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    base_path = r"C:\Users\LENOVO\Desktop\Segmentation paper DG\data\Cataract-1K\coco\padded_splited_data"
    train_json = os.path.join(base_path, "labels", "train_annotations.json")
    train_img_dir = os.path.join(base_path, "images", "train")

    num_classes = 13
    
    # استفاده از دیتاست
    dataset = SurgicalDataset(train_img_dir, train_json, transforms=get_transform(train=True))
    
    # ۲. ارجاع به تابع تعریف شده به جای استفاده از lambda
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=2, # در ویندوز اعداد بالا مثل 4 گاهی ارور می‌دهند، با 2 شروع کنید
        collate_fn=collate_fn 
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # ... بقیه کد آموزش (Optimizer و Loop)
    print("🚀 Setup complete. Starting training loop...")
    # کدهای قبلی آموزش را اینجا ادامه دهید
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print("@")

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}: Loss {losses.item()}")

if __name__ == "__main__":
    main()


