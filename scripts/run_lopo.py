"""
Leave-one-patient-out runner across 4 conditions:
  - linear  / no augmentation
  - linear  / augmentation
  - adapter / no augmentation
  - adapter / augmentation

For each of the 6 slides, that slide is held out as the test fold, one other
slide acts as val (deterministic rotation -- see ROTATION below), and the
remaining 4 slides train. Total = 6 folds * 4 conditions = 24 runs.

Per-run outputs:
  outputs/lopo/<condition>/heldout_<slide>/
    config.yaml          # the exact fold config that was used
    predictions.csv      # per-image predictions on the held-out slide
    checkpoints/         # best-by-val-acc checkpoint for this fold

Aggregate output:
  outputs/lopo/summary.csv  # one row per (condition, held-out slide) with
                            # accuracy / precision / recall / F1 / counts

Run with:
  python scripts/run_lopo.py --config config/default.yaml
or to run just one condition:
  python scripts/run_lopo.py --conditions linear_noaug
"""

import argparse
import copy
import csv
import gc
import sys
from pathlib import Path

# Make the package importable whether invoked as `python scripts/run_lopo.py`
# or `python -m scripts.run_lopo` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.pathology_poc.cli import train_run, evaluate_run, load_config


# Each row: (test_slide, val_slide, train_slides).
# 4-slide rotation across s5, s6, s7, s8. Every slide plays every role:
# test 1x, val 1x, train 2x. Slides s1 and s2 are excluded from this sweep
# (insufficient crops -- noted as a collection limitation).
ROTATION = [
    ("slide_s5", "slide_s6", ["slide_s7", "slide_s8"]),
    ("slide_s6", "slide_s7", ["slide_s5", "slide_s8"]),
    ("slide_s7", "slide_s8", ["slide_s5", "slide_s6"]),
    ("slide_s8", "slide_s5", ["slide_s6", "slide_s7"]),
]

# (condition_name, model.mode, data.augment)
CONDITIONS = [
    ("linear_noaug",  "linear",  False),
    ("linear_aug",    "linear",  True),
    ("adapter_noaug", "adapter", False),
    ("adapter_aug",   "adapter", True),
]


def build_fold_config(base_cfg, mode, augment, train_slides, val_slide, test_slide, out_root):
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("model", {})["mode"] = mode
    data_cfg = cfg.setdefault("data", {})
    data_cfg["augment"] = augment
    data_cfg["splits"] = {
        "train": list(train_slides),
        "val": [val_slide],
        "test": [test_slide],
        "hard_negative": [],
    }
    output_cfg = cfg.setdefault("output", {})
    output_cfg["checkpoint_dir"] = str(out_root / "checkpoints")
    output_cfg["inference_csv"] = str(out_root / "predictions.csv")
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="config/default.yaml",
                    help="Base config; data.splits / model.mode / data.augment / "
                         "output paths are overridden per fold")
    ap.add_argument("--output-root", default="outputs/lopo")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Subset of conditions to run, e.g. "
                         "--conditions linear_noaug adapter_aug")
    ap.add_argument("--slides", nargs="*", default=None,
                    help="Subset of held-out slides, e.g. --slides slide_s1 slide_s2")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    selected_conditions = CONDITIONS
    if args.conditions:
        selected_conditions = [c for c in CONDITIONS if c[0] in args.conditions]
        if not selected_conditions:
            ap.error(f"No matching conditions in {args.conditions}; "
                     f"available: {[c[0] for c in CONDITIONS]}")

    selected_folds = ROTATION
    if args.slides:
        selected_folds = [f for f in ROTATION if f[0] in args.slides]
        if not selected_folds:
            ap.error(f"No matching held-out slides in {args.slides}; "
                     f"available: {[f[0] for f in ROTATION]}")

    summary_rows = []
    n_runs = len(selected_conditions) * len(selected_folds)
    run_idx = 0

    for cond_name, mode, augment in selected_conditions:
        for test_slide, val_slide, train_slides in selected_folds:
            run_idx += 1
            tag = f"{cond_name}/heldout_{test_slide}"
            fold_dir = out_root / cond_name / f"heldout_{test_slide}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n========== [{run_idx}/{n_runs}] {tag} ==========")
            print(f"  mode={mode}  augment={augment}")
            print(f"  train: {train_slides}")
            print(f"  val:   [{val_slide}]")
            print(f"  test:  [{test_slide}]")

            cfg = build_fold_config(
                base_cfg, mode, augment, train_slides, val_slide, test_slide, fold_dir,
            )

            # Persist the fold config alongside its outputs for traceability.
            with open(fold_dir / "config.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            ckpt_path = train_run(cfg)
            if ckpt_path is None:
                print(f"[skip] {tag}: training produced no checkpoint")
                _free_gpu()
                continue

            metrics = evaluate_run(
                cfg, ckpt_path, split="test", output=cfg["output"]["inference_csv"],
            )
            if metrics is None:
                print(f"[skip] {tag}: eval failed")
                _free_gpu()
                continue

            summary_rows.append({
                "condition": cond_name,
                "mode": mode,
                "augment": augment,
                "heldout_slide": test_slide,
                "val_slide": val_slide,
                "train_slides": "|".join(train_slides),
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "checkpoint": ckpt_path,
                "predictions_csv": cfg["output"]["inference_csv"],
            })

            _free_gpu()

    summary_path = out_root / "summary.csv"
    fieldnames = [
        "condition", "mode", "augment", "heldout_slide", "val_slide",
        "train_slides", "n", "accuracy", "precision", "recall", "f1",
        "tp", "fp", "fn", "tn", "checkpoint", "predictions_csv",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"\n========== DONE ==========")
    print(f"Summary: {summary_path}  ({len(summary_rows)}/{n_runs} runs)")


def _free_gpu():
    """Drop cached CUDA memory between folds so we don't leak across runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
