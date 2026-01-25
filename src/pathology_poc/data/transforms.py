from torchvision import transforms

def make_eval_transform(img_size: int = 518):
    """Deterministic transform for eval/inference.
    You can swap this with timm.create_transform to match the exact DINOv2 cfg later.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        # ImageNet/DINOv2-style normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
