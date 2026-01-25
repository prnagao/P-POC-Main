import argparse
import os
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# ✅ reuse the working loader from infer_blind.py
from .infer_blind import load_model


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with full FOV images (100x).",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pth).",
    )
    ap.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Where to save FOV-level predictions.",
    )
    ap.add_argument(
        "--use_adapter",
        action="store_true",
        help="Force use of AdapterClassifier (otherwise inferred from ckpt).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Neutrophil presence threshold on max probability.",
    )
    return ap.parse_args()


def tile_fov(img_tensor, patch_size=518, stride=518):
    """
    img_tensor: (3, H, W)
    Returns:
      patches: (N, 3, patch_size, patch_size) or None
      coords:  list of (x, y) top-left coordinates for each patch
    """
    _, H, W = img_tensor.shape
    patches = []
    coords = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img_tensor[:, y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coords.append((x, y))

    if not patches:
        return None, None

    patches = torch.stack(patches, dim=0)  # (N, 3, patch_size, patch_size)
    return patches, coords


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # IMPORTANT: match training normalization, but no resize/crop
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # adjust if you changed these in training
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # ✅ Build model + load weights using the same helper as infer_blind.py
    model, meta = load_model(args.checkpoint, args.use_adapter, device)
    class_names = meta.get("class_names", ["neutrophil", "rbc"])

    # Figure out which index is "neutrophil"
    try:
        neut_idx = class_names.index("neutrophil")
    except ValueError:
        # Fallback: assume index 0 if name not found
        neut_idx = 0
        print(
            "[infer_fov_presence] WARNING: 'neutrophil' not found in class_names; "
            "defaulting to index 0."
        )

    print(
        f"[infer_fov_presence] Loaded checkpoint {meta.get('checkpoint_name', '')} "
        f"with classes={class_names}, neutrophil index={neut_idx}"
    )

    model.eval()

    input_dir = Path(args.input_dir)
    fov_paths = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
        ]
    )

    if not fov_paths:
        print(f"[infer_fov_presence] No images found in {input_dir}")
        return

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fov_name", "neut_present", "max_p_neut"])

        for fov_path in fov_paths:
            img = Image.open(fov_path).convert("RGB")
            img_tensor = transform(img)  # (3, H, W)

            patches, coords = tile_fov(img_tensor, patch_size=518, stride=518)
            if patches is None:
                writer.writerow([fov_path.name, 0, 0.0])
                continue

            patches = patches.to(device)
            with torch.no_grad():
                logits = model(patches)  # (N, num_classes)
                probs = torch.softmax(logits, dim=1)

            # ✅ use the correct neutrophil index from class_names
            p_neut = probs[:, neut_idx].cpu().numpy()
            max_p_neut = float(p_neut.max())
            neut_present = int(max_p_neut >= args.threshold)

            writer.writerow([fov_path.name, neut_present, max_p_neut])

    print(f"[infer_fov_presence] Wrote FOV-level predictions to {out_path}")


if __name__ == "__main__":
    main()
