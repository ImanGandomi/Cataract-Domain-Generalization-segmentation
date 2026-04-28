import torchvision.transforms.functional as F

class Compose:
    """
    ترکیب ترنسفرم‌ها برای مدیریت همزمان تصویر و هدف
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    """
    تبدیل PIL Image به Tensor
    """
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform(train):
    # در این مرحله، چه برای Train و چه برای Val، فقط ToTensor را اجرا می‌کنیم
    transforms = [ToTensor()]
    return Compose(transforms)