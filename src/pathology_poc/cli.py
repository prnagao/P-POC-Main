import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from rich import print as rprint
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from .data.datasets import build_dataloaders_from_config
from .models.dinov2 import DinoV2Classifier, count_trainable_params
from .models.adapter import AdapterClassifier
from .utils import device, seed_everything


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_mode(cfg: Dict) -> str:
    return str(cfg.get("model", {}).get("mode", "linear")).lower().strip()


def infer_checkpoint_mode(state_dict: Dict) -> str:
    # Best-effort inference for older checkpoints that don't store model_kind.
    keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
    if any("adapter" in k.lower() for k in keys):
        return "adapter"
    return "linear"


def resolve_positive_class_index(class_names: List[str], cfg: Dict) -> int:
    positive_name = cfg.get("data", {}).get("positive_class") or class_names[0]
    if positive_name not in class_names:
        raise ValueError(f"Positive class '{positive_name}' not found in class_names: {class_names}")
    return class_names.index(positive_name)


def prepare_model(cfg: Dict, num_classes: int) -> torch.nn.Module:
    model_cfg = cfg["model"]
    mode = get_model_mode(cfg)
    img_size = cfg["data"]["img_size"]

    if mode == "adapter":
        model = AdapterClassifier(
            num_classes=num_classes,
            freeze_backbone=True,
            model_name=model_cfg["backbone"],
            use_hub=model_cfg.get("use_hub", False),
            hub_name=model_cfg.get("hub_name"),
            use_registers=model_cfg.get("use_registers", False),
            img_size=img_size,
            adapter_hidden_ratio=model_cfg.get("adapter_hidden_ratio", 0.25),
            adapter_dropout=model_cfg.get("adapter_dropout", 0.1),
            adapter_use_layernorm=model_cfg.get("adapter_use_layernorm", True),
        )
    else:
        model = DinoV2Classifier(
            num_classes=num_classes,
            freeze_backbone=True,
            model_name=model_cfg["backbone"],
            use_hub=model_cfg.get("use_hub", False),
            hub_name=model_cfg.get("hub_name"),
            use_registers=model_cfg.get("use_registers", False),
            img_size=img_size,
        )

    return model.to(device())


def save_checkpoint(model: torch.nn.Module, class_names: List[str], cfg: Dict, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)

    mode = get_model_mode(cfg)
    prefix = "adapter" if mode == "adapter" else "linear_probe"
    ckpt_name = f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.pth"
    ckpt_path = os.path.join(save_dir, ckpt_name)

    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "cfg": cfg,
            "model_kind": mode,  # prevents evaluating adapter ckpt with linear model (and vice versa)
        },
        ckpt_path,
    )
    return ckpt_path


def train(args):
    cfg = load_config(args.config)
    seed_everything(cfg["train"]["seed"])

    loaders, class_names = build_dataloaders_from_config(cfg, splits=["train", "val"])
    if "train" not in loaders:
        rprint("[red]No training data found for the requested slide splits.[/red]")
        return

    mode = get_model_mode(cfg)
    rprint(f"[bold]Model mode:[/bold] {mode}")

    model = prepare_model(cfg, num_classes=len(class_names))
    total, trainable = count_trainable_params(model)
    rprint(f"[bold]Parameters:[/bold] total={total:,} trainable={trainable:,} (backbone frozen)")

    # Only optimize parameters that will actually receive gradients.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        rprint("[red]No trainable parameters found. Did you accidentally freeze everything?[/red]")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=device() == "cuda")

    best_val_acc = -1.0
    best_ckpt_path: Optional[str] = None

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}")
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device(), non_blocking=True)
            labels = labels.to(device(), non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device() == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Avoid PyTorch warning about converting tensors that require grad to python scalars.
            pbar.set_postfix(loss=loss.detach().item())

        # validation accuracy
        if "val" in loaders:
            model.eval()
            correct = total_samples = 0
            with torch.no_grad():
                for imgs, labels, _ in loaders["val"]:
                    imgs = imgs.to(device(), non_blocking=True)
                    labels = labels.to(device(), non_blocking=True)
                    logits = model(imgs)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total_samples += labels.numel()

            val_acc = correct / total_samples if total_samples else 0.0
            rprint(f"[cyan]Val accuracy:[/cyan] {val_acc:.3f}")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_ckpt_path = save_checkpoint(model, class_names, cfg, cfg["output"]["checkpoint_dir"])
                rprint(f"[green]Saved checkpoint:[/green] {best_ckpt_path}")

    # If there's no val split, still save something useful.
    if "val" not in loaders:
        best_ckpt_path = save_checkpoint(model, class_names, cfg, cfg["output"]["checkpoint_dir"])
        rprint(f"[green]Saved checkpoint (no val split):[/green] {best_ckpt_path}")

    rprint(f"[bold]Training finished. Best val accuracy:[/bold] {max(best_val_acc, 0.0):.3f}")
    if best_ckpt_path:
        rprint(f"[bold]Best checkpoint:[/bold] {best_ckpt_path}")


def collect_predictions(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, class_names: List[str], positive_idx: int
):
    rows = []
    model.eval()
    with torch.no_grad():
        for imgs, labels, meta in loader:
            logits = model(imgs.to(device(), non_blocking=True))
            probs = torch.softmax(logits, dim=1).cpu()
            pos_probs = probs[:, positive_idx]
            preds = probs.argmax(dim=1)
            for i in range(len(labels)):
                rows.append(
                    {
                        "filename": meta["filename"][i],
                        "slide_id": meta["slide_id"][i],
                        "true_label": meta["true_label"][i],
                        "predicted_probability": float(pos_probs[i].item()),
                        "predicted_class": class_names[preds[i].item()],
                    }
                )
    return rows


def save_csv(rows: List[Dict], path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    fieldnames = ["filename", "slide_id", "true_label", "predicted_probability", "predicted_class"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_checkpoint_compatibility(ckpt: Dict, cfg: Dict, class_names: List[str]) -> bool:
    cfg_mode = get_model_mode(cfg)
    state_dict = ckpt.get("model_state", {})
    ckpt_mode = ckpt.get("model_kind") or infer_checkpoint_mode(state_dict)

    if ckpt_mode != cfg_mode:
        rprint(
            f"[red]Checkpoint/model mismatch:[/red] checkpoint looks like '{ckpt_mode}' "
            f"but config model.mode is '{cfg_mode}'."
        )
        rprint("[red]Fix:[/red] set model.mode to match the checkpoint OR evaluate a checkpoint trained in this mode.")
        return False

    ckpt_class_names = ckpt.get("class_names")
    if ckpt_class_names and ckpt_class_names != class_names:
        rprint("[red]Checkpoint/config mismatch:[/red] checkpoint class_names != config class_names")
        rprint(f"  ckpt:   {ckpt_class_names}")
        rprint(f"  config: {class_names}")
        return False

    return True


def evaluate(args):
    cfg = load_config(args.config)
    loaders, class_names = build_dataloaders_from_config(cfg, splits=[args.split])
    if args.split not in loaders:
        rprint(f"[red]No data found for split '{args.split}'.[/red]")
        return

    ckpt = torch.load(args.checkpoint, map_location=device())
    if not validate_checkpoint_compatibility(ckpt, cfg, class_names):
        return

    model = prepare_model(cfg, num_classes=len(class_names))

    # Prefer strict=True to catch silent mismatch early; fall back only if needed.
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        rprint(f"[yellow]Warning:[/yellow] strict checkpoint load failed; retrying with strict=False")
        rprint(f"[yellow]{e}[/yellow]")
        model.load_state_dict(ckpt["model_state"], strict=False)

    positive_idx = resolve_positive_class_index(class_names, cfg)
    rows = collect_predictions(model, loaders[args.split], class_names, positive_idx)

    # basic metrics (precision/recall/F1) for interpretability
    y_true = [class_names.index(r["true_label"]) for r in rows]
    y_pred = [class_names.index(r["predicted_class"]) for r in rows]
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == positive_idx)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive_idx and yp == positive_idx)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_idx and yp != positive_idx)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    rprint(f"[bold]Precision:[/bold] {precision:.3f}  [bold]Recall:[/bold] {recall:.3f}  [bold]F1:[/bold] {f1:.3f}")

    output_csv = args.output or cfg["output"].get("inference_csv", "outputs/predictions.csv")
    save_csv(rows, output_csv)
    rprint(f"[green]Saved per-image predictions to[/green] {output_csv}")


def infer(args):
    cfg = load_config(args.config)
    loaders, class_names = build_dataloaders_from_config(cfg, splits=[args.split])
    if args.split not in loaders:
        rprint(f"[red]No data found for split '{args.split}'.[/red]")
        return

    ckpt = torch.load(args.checkpoint, map_location=device())
    if not validate_checkpoint_compatibility(ckpt, cfg, class_names):
        return

    model = prepare_model(cfg, num_classes=len(class_names))

    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        rprint(f"[yellow]Warning:[/yellow] strict checkpoint load failed; retrying with strict=False")
        rprint(f"[yellow]{e}[/yellow]")
        model.load_state_dict(ckpt["model_state"], strict=False)

    positive_idx = resolve_positive_class_index(class_names, cfg)
    rows = collect_predictions(model, loaders[args.split], class_names, positive_idx)
    output_csv = args.output or cfg["output"].get("inference_csv", "outputs/predictions.csv")
    save_csv(rows, output_csv)
    rprint(f"[green]Saved per-image predictions to[/green] {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Slide-aware DINOv2 PoC (linear probe or adapter)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train (linear probe or adapter; choose via config model.mode)")
    train_p.add_argument("--config", type=str, default="config/default.yaml")
    train_p.set_defaults(func=train)

    eval_p = subparsers.add_parser("eval", help="Evaluate a checkpoint and emit per-image metrics + CSV")
    eval_p.add_argument("--config", type=str, default="config/default.yaml")
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate (train/val/test/hard_negative)"
    )
    eval_p.add_argument("--output", type=str, help="Optional override for CSV path")
    eval_p.set_defaults(func=evaluate)

    infer_p = subparsers.add_parser("infer", help="Run inference on a configured split and emit a CSV")
    infer_p.add_argument("--config", type=str, default="config/default.yaml")
    infer_p.add_argument("--checkpoint", type=str, required=True)
    infer_p.add_argument("--split", type=str, default="test", help="Dataset split to use (train/val/test/hard_negative)")
    infer_p.add_argument("--output", type=str, help="Optional override for CSV path")
    infer_p.set_defaults(func=infer)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
