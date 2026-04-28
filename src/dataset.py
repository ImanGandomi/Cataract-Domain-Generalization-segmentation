import numpy as np
import torch
import os
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms.functional as F

class SurgicalDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objs = len(anns)
        boxes = []
        labels = []
        masks = []
        
        for i in range(num_objs):
            xmin, ymin, w, h = anns[i]['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(anns[i]['category_id']) 
            masks.append(coco.annToMask(anns[i]))

        # --- رفع UserWarning: استفاده از np.stack برای سرعت بیشتر ---
        if num_objs > 0:
            masks = np.stack(masks, axis=0)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # مدیریت حالتی که هیچ اشیایی در تصویر نیست
            masks = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if num_objs > 0 else torch.tensor([0]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        # --- رفع AttributeError: تبدیل تصویر به تنسور ---
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            # اگر ترنسفرمی تعریف نشده، حداقل تصویر را به تنسور تبدیل کن
            img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.ids)