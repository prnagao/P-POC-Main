import argparse, os, yaml
from pathlib import Path
from typing import List
import torch
from PIL import Image
from torchvision import transforms
from rich import print as rprint

from .models.dinov2 import DinoV2Classifier
from .utils import device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def parse_args():
    p = argparse.ArgumentParser(description="Run inference on one image or a folder")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=False, help="Path to a trained .pth checkpoint")
    p.add_argument("--image", type=str, help="Path to a single image")
    p.add_argument("--folder", type=str, help="Path to a folder of images")
    return p.parse_args()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def gather_images(folder: str) -> List[str]:
    paths = []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(str(p))
    return sorted(paths)

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    if not (args.image or args.folder):
        rprint("[red]Provide --image or --folder[/red]")
        return

    class_names = cfg.get("class_names", ["neutrophil", "rbc"])
    model = DinoV2Classifier(num_classes=len(class_names), freeze_backbone=False, model_name=cfg["model_name"],
                             use_hub=cfg.get("use_hub", False), hub_name=cfg.get("hub_name", None),
                             use_registers=cfg.get("use_registers", False), img_size=cfg["img_size"]).to(device())

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device())
        model.load_state_dict(ckpt["model_state"], strict=False)
        class_names = ckpt.get("class_names", class_names)
        rprint(f"[green]Loaded checkpoint[/green] {args.checkpoint}")
    else:
        rprint("[yellow]No checkpoint provided. Using untrained weights (predictions will be random).[/yellow]")

    model.eval()
    tfm = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    paths = [args.image] if args.image else gather_images(args.folder)
    if not paths:
        rprint("[red]No images found.[/red]")
        return

    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device())
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = probs.argmax()
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
        rprint(f"{p} -> [bold]{pred_name}[/bold]  (probs={probs})")

if __name__ == "__main__":
    main()
