import argparse
import csv
import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image

from .models.dinov2 import DinoV2Classifier
from .models.adapter import AdapterClassifier
from .data.transforms import make_eval_transform


def list_images(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def load_model(
    checkpoint: str,
    use_adapter: bool,
    device: str,
):
    """
    Build the *same* model type used in training (DinoV2Classifier or AdapterClassifier)
    and load the 'model_state' from your checkpoint.
    """
    state = torch.load(checkpoint, map_location=device)

    # Class names + config come from the training script
    class_names = state.get("class_names", ["neutrophil", "rbc"])
    cfg = state.get("cfg", {}) or {}
    model_name = cfg.get("model_name", "vit_small_patch14_dinov2.lvd142m")
    use_hub = cfg.get("use_hub", False)
    hub_name = cfg.get("hub_name", None)
    use_registers = cfg.get("use_registers", False)
    img_size = cfg.get("img_size", 518)

    ckpt_kind = state.get("model_kind", "baseline")

    # Decide which class to instantiate
    if use_adapter or ckpt_kind == "adapter":
        model = AdapterClassifier(
            num_classes=len(class_names),
            model_name=model_name,
            use_hub=use_hub,
            hub_name=hub_name,
            use_registers=use_registers,
            img_size=img_size,
        )
        model_kind = "adapter"
    else:
        model = DinoV2Classifier(
            num_classes=len(class_names),
            freeze_backbone=False,  # freeze is irrelevant at inference
            model_name=model_name,
            use_hub=use_hub,
            hub_name=hub_name,
            use_registers=use_registers,
            img_size=img_size,
        )
        model_kind = "baseline"

    # Load the actual trained weights
    state_dict = state.get("model_state", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[infer_blind] Warning: non-strict load. "
            f"Missing={len(missing)}, Unexpected={len(unexpected)}"
        )
        if missing:
            print("  Missing keys (first 5):", missing[:5])
        if unexpected:
            print("  Unexpected keys (first 5):", unexpected[:5])

    model.to(device).eval()

    meta = {
        "model_kind": model_kind,
        "checkpoint_name": Path(checkpoint).name,
        "backbone": model_name,
        "img_size": img_size,
        "class_names": class_names,
    }
    return model, meta


def main():
    ap = argparse.ArgumentParser(
        description="Blind, human-verifiable inference (no labels)."
    )
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Folder with unlabeled images (recurses).",
    )
    ap.add_argument(
        "--output_csv",
        required=True,
        help="Where to write per-image predictions.",
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Trained checkpoint (.pth) from train/adapter_train.",
    )
    ap.add_argument(
        "--class_names",
        nargs="+",
        help="Optional override for class order, e.g. neutrophil rbc.",
    )
    ap.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Optional override for image size (defaults to checkpoint cfg).",
    )
    ap.add_argument(
        "--use_adapter",
        action="store_true",
        help="If set, force use of AdapterClassifier.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model that matches the training scaffold
    model, meta = load_model(args.checkpoint, args.use_adapter, device)

    # Class names: CLI override if provided & length matches
    class_names = meta["class_names"]
    if args.class_names:
        if len(args.class_names) != len(class_names):
            print(
                "[infer_blind] WARNING: --class_names length does not match "
                "checkpoint; using checkpoint order."
            )
        else:
            class_names = args.class_names

    # Image size: CLI override if provided, else use cfg from checkpoint
    img_size = args.img_size or meta["img_size"]
    transform = make_eval_transform(img_size)

    files = list_images(Path(args.input_dir))
    if not files:
        print("[infer_blind] No images found in", args.input_dir)
        return

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().isoformat(timespec="seconds")
    fieldnames = (
        ["filepath", "pred_label"]
        + [f"prob_{c}" for c in class_names]
        + ["max_prob", "model_kind", "checkpoint_name", "backbone", "img_size", "timestamp"]
    )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for p in files:
                img = Image.open(p).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0].cpu()
                pred_idx = int(probs.argmax().item())
                pred_label = class_names[pred_idx]

                row = {
                    "filepath": str(p),
                    "pred_label": pred_label,
                    **{
                        f"prob_{c}": float(probs[i].item())
                        for i, c in enumerate(class_names)
                    },
                    "max_prob": float(probs.max().item()),
                    "model_kind": meta["model_kind"],
                    "checkpoint_name": meta["checkpoint_name"],
                    "backbone": meta["backbone"],
                    "img_size": img_size,
                    "timestamp": now,
                }
                writer.writerow(row)

    print(f"[infer_blind] Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
