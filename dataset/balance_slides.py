"""
Per-slide 50/50 class balancing for slide-aware training.

Why this exists, and how it differs from Standardize_dataset.py:
  - Standardize_dataset.py pools all slides in a split, then undersamples
    classes globally. That breaks the per-slide folder structure and can
    produce folds where one slide contributes only positives (or only
    negatives) to the pool, which confounds a "generalizes across patients"
    claim.
  - This script balances each slide *independently*: for each slide,
    k = min(#positives, #negatives), and k randomly-sampled images from each
    class are kept. The slide-as-folder layout is preserved, which is what
    SlideCropDataset expects.
  - The dropped images are moved (or copied/symlinked) to an "unused" tree so
    nothing is destroyed and you can re-balance with a different seed later.

Input layout:
  raw_root/
    Slide_s1/
      Positives/   (case-insensitive: positive, pos, p also accepted)
      Negatives/   (case-insensitive: negative, neg, n also accepted)
    Slide_s2/
      ...

Output layout (matches SlideCropDataset):
  output_root/
    <slide>/positive/<file>
    <slide>/negative/<file>
    balance_manifest.csv

  unused_root/
    <slide>/positive/<file>
    <slide>/negative/<file>
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

POS_NAMES = {"positives", "positive", "pos", "p"}
NEG_NAMES = {"negatives", "negative", "neg", "n"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def find_class_folder(slide_dir: Path, names: set):
    for child in slide_dir.iterdir():
        if child.is_dir() and child.name.lower() in names:
            return child
    return None


def list_images(folder):
    if folder is None or not folder.exists():
        return []
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )


def place(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def scan(input_root: Path):
    slides = []
    for entry in sorted(input_root.iterdir()):
        if not entry.is_dir():
            continue
        pos_dir = find_class_folder(entry, POS_NAMES)
        neg_dir = find_class_folder(entry, NEG_NAMES)
        if pos_dir is None and neg_dir is None:
            continue
        slides.append((entry.name, pos_dir, neg_dir))
    return slides


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", "-i", default=".",
                    help="Root containing per-slide folders (default: cwd)")
    ap.add_argument("--output", "-o", default="./dataset_balanced",
                    help="Where balanced, kept images go")
    ap.add_argument("--unused", "-u", default="./dataset_unused",
                    help="Where dropped images go")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["copy", "symlink", "move"], default="copy",
                    help="How to relocate files. 'copy' is safest; 'move' is "
                         "destructive but compact; 'symlink' avoids duplication.")
    ap.add_argument("--min-k", type=int, default=0,
                    help="Skip slides where k = min(#pos, #neg) is below this")
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="Cap kept images at exactly N per class per slide. "
                         "Slides that don't have at least N of each class are skipped. "
                         "Without this flag, each slide keeps min(#pos, #neg).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan, write nothing")
    args = ap.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    unused_root = Path(args.unused).resolve()
    rng = random.Random(args.seed)

    slides = scan(input_root)
    if not slides:
        ap.error(f"No slide folders found under {input_root}")

    print(f"Scanning: {input_root}")
    print(f"Found {len(slides)} slide(s):\n")

    rows = []
    for slide_name, pos_dir, neg_dir in slides:
        pos = list_images(pos_dir)
        neg = list_images(neg_dir)
        k = min(len(pos), len(neg))

        if k == 0:
            print(f"  [skip] {slide_name}: {len(pos)} pos, {len(neg)} neg "
                  f"(zero of one class)")
            rows.append({
                "slide": slide_name, "n_pos": len(pos), "n_neg": len(neg),
                "k": 0, "kept_pos": 0, "kept_neg": 0,
                "dropped_pos": len(pos), "dropped_neg": len(neg),
                "status": "skipped_empty_class",
            })
            continue

        if k < args.min_k:
            print(f"  [skip] {slide_name}: k={k} below --min-k={args.min_k}")
            rows.append({
                "slide": slide_name, "n_pos": len(pos), "n_neg": len(neg),
                "k": k, "kept_pos": 0, "kept_neg": 0,
                "dropped_pos": len(pos), "dropped_neg": len(neg),
                "status": "skipped_min_k",
            })
            continue

        # If --n-per-class is set, every kept slide gets exactly that many per
        # class. Slides without enough of either class are dropped entirely so
        # fold sizes stay uniform across the experiment.
        if args.n_per_class is not None:
            if len(pos) < args.n_per_class or len(neg) < args.n_per_class:
                print(f"  [skip] {slide_name}: needs >= {args.n_per_class} "
                      f"of each class, has {len(pos)} pos / {len(neg)} neg")
                rows.append({
                    "slide": slide_name, "n_pos": len(pos), "n_neg": len(neg),
                    "k": k, "kept_pos": 0, "kept_neg": 0,
                    "dropped_pos": len(pos), "dropped_neg": len(neg),
                    "status": "skipped_below_n_per_class",
                })
                continue
            keep_n = args.n_per_class
        else:
            keep_n = k

        pos_shuffled = list(pos); rng.shuffle(pos_shuffled)
        neg_shuffled = list(neg); rng.shuffle(neg_shuffled)
        pos_keep, pos_drop = pos_shuffled[:keep_n], pos_shuffled[keep_n:]
        neg_keep, neg_drop = neg_shuffled[:keep_n], neg_shuffled[keep_n:]

        print(f"  {slide_name}: pos {len(pos)}->{len(pos_keep)}, "
              f"neg {len(neg)}->{len(neg_keep)} (k={k}, dropped "
              f"{len(pos_drop)}+{len(neg_drop)})")

        rows.append({
            "slide": slide_name, "n_pos": len(pos), "n_neg": len(neg),
            "k": k, "kept_pos": len(pos_keep), "kept_neg": len(neg_keep),
            "dropped_pos": len(pos_drop), "dropped_neg": len(neg_drop),
            "status": "balanced",
        })

        if args.dry_run:
            continue

        for src in pos_keep:
            place(src, output_root / slide_name / "positive" / src.name, args.mode)
        for src in neg_keep:
            place(src, output_root / slide_name / "negative" / src.name, args.mode)
        for src in pos_drop:
            place(src, unused_root / slide_name / "positive" / src.name, args.mode)
        for src in neg_drop:
            place(src, unused_root / slide_name / "negative" / src.name, args.mode)

    kept = sum(r["kept_pos"] + r["kept_neg"] for r in rows)
    dropped = sum(r["dropped_pos"] + r["dropped_neg"] for r in rows)
    print(f"\nTotals: kept {kept}, relocated-to-unused {dropped}")

    if args.dry_run:
        print("\n[dry-run] Nothing written. Re-run without --dry-run to commit.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "balance_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "slide", "n_pos", "n_neg", "k",
            "kept_pos", "kept_neg", "dropped_pos", "dropped_neg", "status",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nManifest: {manifest}")
    print(f"Balanced: {output_root}")
    print(f"Unused:   {unused_root}")


if __name__ == "__main__":
    main()
