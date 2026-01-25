import argparse, yaml, torch
from rich import print as rprint
from .data.datasets import build_dataloaders
from .models.dinov2 import DinoV2Classifier
from .utils import device

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained baseline checkpoint on test set")
    p.add_argument("--data_root", type=str, default="dataset")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    loaders, class_names = build_dataloaders(args.data_root, cfg["img_size"], cfg["batch_size"], cfg["num_workers"])
    num_classes = len(class_names)
    if num_classes == 0:
        rprint("[red]No images found in dataset folders.[/red]")
        return

    ckpt = torch.load(args.checkpoint, map_location=device())
    model = DinoV2Classifier(num_classes=num_classes, freeze_backbone=True, model_name=cfg["model_name"],
                             use_hub=cfg.get("use_hub", False), hub_name=cfg.get("hub_name", None),
                             use_registers=cfg.get("use_registers", False), img_size=cfg["img_size"]).to(device())
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loaders["test"]:
            if imgs.shape[0] == 0: continue
            imgs = imgs.to(device())
            labels = labels.to(device())
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    if total == 0:
        rprint("[yellow]No test images to evaluate.[/yellow]")
    else:
        acc = correct / total
        rprint(f"[bold]Test accuracy (baseline):[/bold] {acc:.3f}  ({correct}/{total})")

if __name__ == "__main__":
    main()
