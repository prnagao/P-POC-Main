"""
External validation on the NIH (Rajaraman 2018) malaria cell images dataset.

Walks each fold checkpoint from `outputs/lopo/<condition>/heldout_<slide>/checkpoints/`,
runs inference over the NIH cell images, writes per-image predictions, and
computes two summary tables:

  - per_checkpoint_summary.csv: one row per (condition, fold). Use this in the
    supplement to show fold-level variance under external shift.
  - per_condition_ensemble_summary.csv: one row per condition, where the 4
    fold checkpoints are ensembled by averaging positive-class probabilities
    before thresholding. Use this in the main table as the headline number.

NIH layout expected (the standard Kaggle / NIH release):

  <nih_root>/
    Parasitized/<file>.png
    Uninfected/<file>.png

Class mapping (default): Parasitized -> positive, Uninfected -> negative.
Override with --pos-folder / --neg-folder if your copy uses different names.

Run:
  python scripts/run_external_nih.py --nih-root path/to/cell_images
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Make the package importable whether invoked from the repo root or scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.pathology_poc.cli import (
    load_config,
    prepare_model,
    resolve_positive_class_index,
)
from src.pathology_poc.data.datasets import VALID_EXTENSIONS, make_eval_transform
from src.pathology_poc.utils import device


class LabeledFolderDataset(Dataset):
    """Flat dataset: <root>/<class_folder>/<image>. No slide grouping."""

    def __init__(self, root, class_name_to_folder, class_names, img_size):
        self.transform = make_eval_transform(img_size)
        self.class_names = class_names
        self.samples = []

        root = Path(root)
        for class_idx, class_name in enumerate(class_names):
            target = class_name_to_folder[class_name].lower()
            actual = None
            for child in root.iterdir():
                if child.is_dir() and child.name.lower() == target:
                    actual = child
                    break
            if actual is None:
                raise FileNotFoundError(
                    f"No folder under {root} matches '{class_name_to_folder[class_name]}' "
                    f"(class '{class_name}'). Subdirectories present: "
                    f"{[c.name for c in root.iterdir() if c.is_dir()]}"
                )
            for f in sorted(actual.rglob("*")):
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((f, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Some NIH PNGs are RGBA or grayscale; force RGB so the transform doesn't choke.
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label, {"filepath": str(path), "filename": path.name,
                              "true_label": self.class_names[label]}


def find_checkpoints(lopo_root):
    """Return {condition: {fold: checkpoint_path}}.

    Picks the most-recent .pth per fold dir, which matches `train_run`'s
    save-on-best-val semantics (the best-by-val checkpoint is also the most
    recently written, because earlier saves are only kept if val acc was beaten)."""
    result = defaultdict(dict)
    lopo_root = Path(lopo_root)
    if not lopo_root.exists():
        return result
    for cond_dir in sorted(lopo_root.iterdir()):
        if not cond_dir.is_dir():
            continue
        for fold_dir in sorted(cond_dir.iterdir()):
            if not fold_dir.is_dir() or not fold_dir.name.startswith("heldout_"):
                continue
            ckpt_dir = fold_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            ckpts = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
            if ckpts:
                result[cond_dir.name][fold_dir.name] = ckpts[-1]
    return result


def confusion(probs, true_labels, positive_idx, threshold):
    """Return TP, FP, FN, TN given pos-class probs, true class indices, threshold."""
    neg_idx = 1 - positive_idx  # binary
    tp = fp = fn = tn = 0
    for prob, true in zip(probs, true_labels):
        pred = positive_idx if prob >= threshold else neg_idx
        if true == positive_idx and pred == positive_idx:
            tp += 1
        elif true == neg_idx and pred == positive_idx:
            fp += 1
        elif true == positive_idx and pred == neg_idx:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def metrics_from_probs(probs, true_labels, positive_idx, threshold):
    tp, fp, fn, tn = confusion(probs, true_labels, positive_idx, threshold)
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0     # == sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "n": n, "accuracy": accuracy, "precision": precision,
        "recall_sensitivity": recall, "specificity": specificity, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def write_predictions_csv(path, filepaths, probs, true_labels, class_names,
                          positive_idx, threshold, prob_col_name):
    neg_idx = 1 - positive_idx
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "filepath", "true_label", prob_col_name, "predicted_class",
        ])
        w.writeheader()
        for fp, prob, lbl in zip(filepaths, probs, true_labels):
            pred = positive_idx if prob >= threshold else neg_idx
            w.writerow({
                "filepath": fp,
                "true_label": class_names[lbl],
                prob_col_name: f"{prob:.6f}",
                "predicted_class": class_names[pred],
            })


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--nih-root", default=None,
                    help="Directory containing Parasitized/ and Uninfected/ subfolders. "
                         "Mutually exclusive with --kaggle.")
    ap.add_argument("--kaggle", action="store_true",
                    help="Auto-download the dataset via kagglehub. Requires "
                         "`pip install kagglehub` and a Kaggle API token at "
                         "~/.kaggle/kaggle.json (Account -> Create New API Token).")
    ap.add_argument("--kaggle-slug", default="iarunava/cell-images-for-detecting-malaria",
                    help="Kaggle dataset slug (only used with --kaggle).")
    ap.add_argument("--pos-folder", default="Parasitized",
                    help="Folder mapped to the positive class (default: Parasitized)")
    ap.add_argument("--neg-folder", default="Uninfected",
                    help="Folder mapped to the negative class (default: Uninfected)")
    ap.add_argument("--lopo-root", default="outputs/lopo")
    ap.add_argument("--output-root", default="outputs/external_nih")
    ap.add_argument("--config", default="config/default.yaml",
                    help="Base config (for class_names, positive_class, img_size)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Subset of conditions, e.g. linear_aug adapter_aug")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="Subset of folds, e.g. heldout_slide_s5 heldout_slide_s7")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    class_names = base_cfg["data"]["class_names"]
    positive_idx = resolve_positive_class_index(class_names, base_cfg)
    img_size = base_cfg["data"]["img_size"]

    folder_map = {class_names[positive_idx]: args.pos_folder,
                  class_names[1 - positive_idx]: args.neg_folder}

    if args.kaggle and args.nih_root:
        ap.error("--kaggle and --nih-root are mutually exclusive")
    if not args.kaggle and not args.nih_root:
        ap.error("Provide either --nih-root or --kaggle")

    if args.kaggle:
        try:
            import kagglehub
        except ImportError:
            ap.error("kagglehub not installed. Run: pip install kagglehub")
        print(f"Downloading via kagglehub: {args.kaggle_slug}")
        downloaded = Path(kagglehub.dataset_download(args.kaggle_slug))
        # The iarunava archive expands to <root>/cell_images/{Parasitized,Uninfected}/,
        # but other slugs may put the class folders directly at <root>. Auto-detect.
        candidate = downloaded / "cell_images"
        nih_root = candidate if candidate.exists() else downloaded
        print(f"  -> {nih_root}")
    else:
        nih_root = args.nih_root

    print(f"Class mapping: {folder_map}")
    print(f"Loading NIH images from: {nih_root}")
    dataset = LabeledFolderDataset(nih_root, folder_map, class_names, img_size)
    print(f"  -> {len(dataset)} images")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    ckpt_map = find_checkpoints(args.lopo_root)
    if not ckpt_map:
        ap.error(f"No checkpoints found under {args.lopo_root}")
    if args.conditions:
        ckpt_map = {c: f for c, f in ckpt_map.items() if c in args.conditions}
    if args.folds:
        ckpt_map = {c: {fold: p for fold, p in folds.items() if fold in args.folds}
                    for c, folds in ckpt_map.items()}

    total = sum(len(folds) for folds in ckpt_map.values())
    print(f"Found {total} checkpoint(s) across {len(ckpt_map)} condition(s)\n")

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    per_ckpt_rows = []
    # cond -> list of per-image prob arrays, one per fold; aligned by sample index.
    per_cond_probs = defaultdict(list)
    file_paths_ref = None
    true_labels_ref = None

    run_idx = 0
    for cond_name, fold_map in ckpt_map.items():
        for fold_name, ckpt_path in fold_map.items():
            run_idx += 1
            tag = f"{cond_name}/{fold_name}"
            print(f"=== [{run_idx}/{total}] {tag} ===")

            ckpt = torch.load(ckpt_path, map_location=device())
            ckpt_cfg = ckpt.get("cfg") or base_cfg
            # The ckpt's stored cfg carries model.mode (linear/adapter), adapter
            # hyperparams, backbone choice -- everything prepare_model needs.
            inference_cfg = dict(base_cfg)
            inference_cfg["model"] = ckpt_cfg.get("model", base_cfg["model"])

            model = prepare_model(inference_cfg, num_classes=len(class_names))
            try:
                model.load_state_dict(ckpt["model_state"], strict=True)
            except RuntimeError as e:
                print(f"  [warn] strict load failed, retrying non-strict: {e}")
                model.load_state_dict(ckpt["model_state"], strict=False)
            model.eval()

            all_probs = []
            all_labels = []
            all_paths = []
            with torch.no_grad():
                for imgs, labels, meta in loader:
                    logits = model(imgs.to(device(), non_blocking=True))
                    probs = torch.softmax(logits, dim=1).cpu()
                    pos = probs[:, positive_idx]
                    all_probs.extend(float(p) for p in pos)
                    all_labels.extend(int(l) for l in labels)
                    all_paths.extend(meta["filepath"])

            m = metrics_from_probs(all_probs, all_labels, positive_idx, args.threshold)
            print(f"  n={m['n']}  acc={m['accuracy']:.3f}  "
                  f"prec={m['precision']:.3f}  sens={m['recall_sensitivity']:.3f}  "
                  f"spec={m['specificity']:.3f}  f1={m['f1']:.3f}")

            fold_out = out_root / cond_name / fold_name
            write_predictions_csv(
                fold_out / "predictions.csv", all_paths, all_probs, all_labels,
                class_names, positive_idx, args.threshold, "prob_positive",
            )

            per_ckpt_rows.append({
                "condition": cond_name,
                "fold": fold_name,
                "checkpoint": str(ckpt_path),
                **m,
            })
            per_cond_probs[cond_name].append(all_probs)

            if file_paths_ref is None:
                file_paths_ref = all_paths
                true_labels_ref = all_labels

            # Free the model before the next fold's backbone loads.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------- Per-checkpoint summary --------
    per_ckpt_path = out_root / "per_checkpoint_summary.csv"
    with open(per_ckpt_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "condition", "fold", "n", "accuracy", "precision",
            "recall_sensitivity", "specificity", "f1",
            "tp", "fp", "fn", "tn", "checkpoint",
        ])
        w.writeheader()
        for r in per_ckpt_rows:
            w.writerow(r)
    print(f"\nPer-checkpoint summary -> {per_ckpt_path}")

    # -------- Per-condition ensemble (average probs across folds) --------
    ensemble_rows = []
    for cond_name, prob_lists in per_cond_probs.items():
        n_folds = len(prob_lists)
        n_images = len(file_paths_ref)
        ensemble_probs = [
            sum(prob_lists[k][i] for k in range(n_folds)) / n_folds
            for i in range(n_images)
        ]
        m = metrics_from_probs(ensemble_probs, true_labels_ref, positive_idx, args.threshold)
        print(f"[ensemble {n_folds}-fold] {cond_name}: "
              f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
              f"sens={m['recall_sensitivity']:.3f}  spec={m['specificity']:.3f}  "
              f"f1={m['f1']:.3f}")

        write_predictions_csv(
            out_root / cond_name / "ensemble_predictions.csv",
            file_paths_ref, ensemble_probs, true_labels_ref,
            class_names, positive_idx, args.threshold, "prob_positive_ensemble",
        )

        ensemble_rows.append({
            "condition": cond_name,
            "n_folds": n_folds,
            **m,
        })

    ensemble_path = out_root / "per_condition_ensemble_summary.csv"
    with open(ensemble_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "condition", "n_folds", "n", "accuracy", "precision",
            "recall_sensitivity", "specificity", "f1",
            "tp", "fp", "fn", "tn",
        ])
        w.writeheader()
        for r in ensemble_rows:
            w.writerow(r)
    print(f"Per-condition ensemble summary -> {ensemble_path}")


if __name__ == "__main__":
    main()
