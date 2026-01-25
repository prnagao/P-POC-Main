from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Optional: for nicer rotation behavior (fill + interpolation).
# If your torchvision is older and doesn't have InterpolationMode, this will fall back safely.
try:
    from torchvision.transforms import InterpolationMode
    _HAS_INTERP = True
except Exception:
    InterpolationMode = None
    _HAS_INTERP = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# -------------------------
# Transforms
# -------------------------

def make_train_transform(img_size: int) -> transforms.Compose:
    """
    Training-only augmentation (per your spec):

    - Random rotation by any angle between 0 and 360 degrees
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random brightness +/- 10–15%  (we use 15% here)
    - Random saturation +/- 10–15%  (we use 15% here)
    - Random hue +/- 10 degrees     (converted to torchvision units)
    - Apply the color adjustment steps to ~75% of training images
    - No augmentations for val/test (handled in make_eval_transform)
    """
    hue_frac = 10.0 / 360.0  # 10 degrees -> torchvision hue fraction

    # Rotation with better defaults if InterpolationMode is available
    if _HAS_INTERP:
        rotate = transforms.RandomRotation(
            degrees=(0, 360),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=255,  # avoids black corner artifacts; change to 0 if you prefer black fill
        )
    else:
        # Older torchvision fallback (no interpolation/fill control)
        rotate = transforms.RandomRotation(degrees=(0, 360))

    color_jitter_group = transforms.RandomApply(
        [
            transforms.ColorJitter(
                brightness=0.15,   # +/- 15%
                saturation=0.15,   # +/- 15%
                hue=hue_frac,      # +/- 10 degrees
            )
        ],
        p=0.75,  # apply color steps to ~75% of training images
    )

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),

            # Geometry aug
            rotate,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            # Color aug (grouped)
            color_jitter_group,

            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_eval_transform(img_size: int) -> transforms.Compose:
    """
    Deterministic transform for validation / test.
    No augmentation.
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# -------------------------
# Dataset
# -------------------------

class SlideCropDataset(Dataset):
    """
    Slide-aware dataset where samples are grouped as:
    slide_id / class_name / image
    """

    def __init__(
        self,
        data_root: str,
        slide_ids: Iterable[str],
        class_names: List[str],
        img_size: int,
        split: str,
    ):
        self.root = Path(data_root)
        self.class_names = class_names
        self.split = split

        # IMPORTANT: train-only augmentation happens here
        if split == "train":
            self.transform = make_train_transform(img_size)
        else:
            self.transform = make_eval_transform(img_size)

        self.samples: List[Tuple[Path, int, str]] = []

        for slide_id in slide_ids:
            slide_dir = self.root / slide_id
            for class_idx, class_name in enumerate(class_names):
                class_dir = slide_dir / class_name
                if not class_dir.exists():
                    continue
                for img_path in class_dir.rglob("*"):
                    if img_path.suffix.lower() not in VALID_EXTENSIONS:
                        continue
                    self.samples.append((img_path, class_idx, slide_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, slide_id = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        metadata = {
            "filepath": str(path),
            "filename": path.name,
            "slide_id": slide_id,
            "true_label": self.class_names[label],
        }
        return image, label, metadata


# -------------------------
# Dataloader builder
# -------------------------

def build_dataloaders_from_config(
    cfg: Dict, splits: Iterable[str]
) -> Tuple[Dict[str, DataLoader], List[str]]:
    data_cfg = cfg["data"]
    class_names = data_cfg["class_names"]
    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        slide_ids = data_cfg.get("splits", {}).get(split, [])
        dataset = SlideCropDataset(
            data_root=data_cfg["root"],
            slide_ids=slide_ids,
            class_names=class_names,
            img_size=data_cfg["img_size"],
            split=split,
        )
        if len(dataset) == 0:
            continue

        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg["train"].get("num_workers", 2),
            pin_memory=torch.cuda.is_available(),
        )

    return loaders, class_names
