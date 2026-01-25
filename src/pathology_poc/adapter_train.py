import argparse, os, time, yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from rich import print as rprint

from .data.datasets import build_dataloaders
from .models.adapter import AdapterClassifier
from .utils import seed_everything, device

def parse_args():
    p = argparse.ArgumentParser(description="Train adapter on top of frozen DINOv2 backbone")
    p.add_argument("--data_root", type=str, default="dataset")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    # Loader options
    p.add_argument("--use_hub", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--hub_name", type=str, default=None)
    p.add_argument("--use_registers", action=argparse.BooleanOptionalAction, default=None)
    # Adapter options
    p.add_argument("--adapter_hidden_ratio", type=float, default=None)
    p.add_argument("--adapter_dropout", type=float, default=None)
    p.add_argument("--adapter_use_layernorm", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--dry-run", action="store_true", help="Build everything & run a dummy forward; no training")
    return p.parse_args()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    # CLI override
    for k in ["img_size","batch_size","epochs","lr","weight_decay","checkpoint_dir","seed",
              "use_hub","hub_name","use_registers",
              "adapter_hidden_ratio","adapter_dropout","adapter_use_layernorm"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    seed_everything(cfg.get("seed", 42))

    loaders, class_names = build_dataloaders(
        data_root=args.data_root,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    num_classes = len(class_names) if class_names else 2
    if num_classes < 2:
        num_classes = 2
        class_names = class_names or ["neutrophil", "rbc"]

    rprint(f"[bold]Classes:[/bold] {class_names}")
    rprint(f"[bold]Image size:[/bold] {cfg['img_size']}  [bold]Batch size:[/bold] {cfg['batch_size']}")

    model = AdapterClassifier(
        num_classes=num_classes,
        freeze_backbone=True,  # adapter exp keeps backbone frozen
        model_name=cfg["model_name"],
        use_hub=cfg.get("use_hub", False),
        hub_name=cfg.get("hub_name", None),
        use_registers=cfg.get("use_registers", False),
        img_size=cfg["img_size"],
        adapter_hidden_ratio=cfg.get("adapter_hidden_ratio", 0.5),
        adapter_dropout=cfg.get("adapter_dropout", 0.1),
        adapter_use_layernorm=cfg.get("adapter_use_layernorm", True),
    ).to(device())

    # Optimizer trains ONLY adapter + head (backbone frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler(enabled=(device()=="cuda"))

    if args.dry_run:
        rprint("[yellow]Dry run (adapter):[/yellow] Creating a dummy batch and running a forward pass...")
        x = torch.randn(cfg["batch_size"], 3, cfg["img_size"], cfg["img_size"]).to(device())
        with torch.no_grad():
            y = model(x)
        rprint(f"[green]OK[/green] Adapter logits shape: {tuple(y.shape)} (B, num_classes)")
        ckpt_path = Path(cfg["checkpoint_dir"]) / "adapter_placeholder.ckpt"
        rprint(f"[bold]Checkpoint target path (when you train):[/bold] {ckpt_path}")
        return

    best_acc = 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"[Adapter] Epoch {epoch+1}/{cfg['epochs']}")
        for imgs, labels in pbar:
            if imgs.shape[0] == 0:  # safeguard in case empty
                continue
            imgs = imgs.to(device(), non_blocking=True)
            labels = labels.to(device(), non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device()=="cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=float(loss))

        # Validation
        model.eval()
        correct = total_samples = 0
        with torch.no_grad():
            for imgs, labels in loaders["val"]:
                if imgs.shape[0] == 0: continue
                imgs = imgs.to(device(), non_blocking=True)
                labels = labels.to(device(), non_blocking=True)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total_samples += labels.numel()
        acc = (correct / total_samples) if total_samples > 0 else 0.0
        rprint(f"[cyan]Val accuracy (adapter):[/cyan] {acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
            ckpt_name = f"adapter_{time.strftime('%Y%m%d-%H%M%S')}_acc{acc:.3f}.pth"
            ckpt_path = os.path.join(cfg["checkpoint_dir"], ckpt_name)
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "cfg": cfg,
                "model_kind": "adapter",
            }, ckpt_path)
            rprint(f"[green]Saved checkpoint:[/green] {ckpt_path}")

    rprint(f"[bold]Best val acc (adapter):[/bold] {best_acc:.3f}")

if __name__ == "__main__":
    main()
