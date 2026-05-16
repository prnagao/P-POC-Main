"""
Post-hoc scoring for a LOPO sweep.

Walks `outputs/lopo/<condition>/heldout_<slide>/predictions.csv` files, then for
each (condition, held-out slide) computes:

  - At threshold = 0.5 (the default operating point used during eval):
      accuracy, precision, recall, F1, TP/FP/FN/TN
  - A precision/recall curve swept across thresholds (0.01..0.99, step 0.01)
      written to outputs/lopo/<condition>/heldout_<slide>/pr_curve.csv
  - The F1-optimal threshold and its precision/recall/F1/specificity/sensitivity
      (sensitivity == recall on the positive class; specificity == TN/(TN+FP))

Aggregates everything into outputs/lopo/scored_summary.csv: one row per
(condition, held-out slide). A second file `scored_per_condition.csv` reports
per-condition mean +/- std across folds for the headline metrics.

Run with:
  python scripts/score_lopo.py --root outputs/lopo
"""

import argparse
import csv
import statistics
from pathlib import Path


THRESHOLDS = [i / 100.0 for i in range(1, 100)]  # 0.01 .. 0.99


def load_predictions(csv_path: Path):
    """Returns (probs, labels) -- both lists, label = 1 for positive class."""
    probs, labels, positive_name = [], [], None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = row["true_label"]
            prob = float(row["predicted_probability"])
            # The "positive class" is whichever class true_label compares against
            # for precision/recall to make sense. The trainer logs probabilities
            # of the positive class, so we infer the positive name from rows
            # where prob > 0.5 matches predicted_class.
            if positive_name is None and row["predicted_class"] != row["true_label"]:
                pass
            probs.append(prob)
            labels.append(true_label)
    # Heuristic: the positive name is the class for which probabilities are
    # written. We can recover it by finding a row where the predicted class
    # equals the positive label by construction. Simpler: read it from a sibling
    # config.yaml if present, otherwise fall back to "positive".
    positive_name = _infer_positive_name(csv_path)
    bin_labels = [1 if l == positive_name else 0 for l in labels]
    return probs, bin_labels, positive_name


def _infer_positive_name(predictions_csv: Path) -> str:
    cfg_path = predictions_csv.parent / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml  # local import so the script still parses without yaml
            cfg = yaml.safe_load(open(cfg_path))
            name = cfg.get("data", {}).get("positive_class")
            if name:
                return str(name)
        except Exception:
            pass
    return "positive"


def confusion_at(probs, labels, thresh):
    tp = fp = fn = tn = 0
    for p, y in zip(probs, labels):
        pred_pos = p >= thresh
        if pred_pos and y == 1:
            tp += 1
        elif pred_pos and y == 0:
            fp += 1
        elif not pred_pos and y == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # == sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "n": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pr_curve(probs, labels):
    rows = []
    for t in THRESHOLDS:
        tp, fp, fn, tn = confusion_at(probs, labels, t)
        m = metrics_from_confusion(tp, fp, fn, tn)
        rows.append({"threshold": round(t, 2), **m})
    return rows


def f1_optimal(pr_rows):
    best = max(pr_rows, key=lambda r: (r["f1"], r["threshold"]))
    return best


def parse_condition_and_slide(predictions_csv: Path):
    # Expect .../<condition>/heldout_<slide>/predictions.csv
    fold_dir = predictions_csv.parent
    cond_dir = fold_dir.parent
    slide = fold_dir.name.replace("heldout_", "")
    condition = cond_dir.name
    return condition, slide


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/lopo",
                    help="Root that contains <condition>/heldout_<slide>/predictions.csv")
    ap.add_argument("--default-threshold", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        ap.error(f"Root does not exist: {root}")

    pred_files = sorted(root.glob("*/heldout_*/predictions.csv"))
    if not pred_files:
        ap.error(f"No predictions.csv files under {root}")

    print(f"Found {len(pred_files)} prediction files.\n")

    per_fold_rows = []
    for pf in pred_files:
        condition, slide = parse_condition_and_slide(pf)
        probs, labels, positive_name = load_predictions(pf)
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        # Metrics at the default operating threshold.
        at_default = metrics_from_confusion(*confusion_at(probs, labels, args.default_threshold))

        # PR sweep and F1-optimal threshold.
        pr_rows = pr_curve(probs, labels)
        pr_csv = pf.parent / "pr_curve.csv"
        with open(pr_csv, "w", newline="") as f:
            fieldnames = ["threshold", "n", "accuracy", "precision", "recall",
                          "sensitivity", "specificity", "f1", "tp", "fp", "fn", "tn"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in pr_rows:
                w.writerow(r)

        best = f1_optimal(pr_rows)

        print(f"{condition} / heldout_{slide}  (n={len(labels)}, "
              f"pos={n_pos}, neg={n_neg}, positive='{positive_name}')")
        print(f"  @0.5:  acc={at_default['accuracy']:.3f}  "
              f"P={at_default['precision']:.3f}  R={at_default['recall']:.3f}  "
              f"F1={at_default['f1']:.3f}")
        print(f"  @F1*  thresh={best['threshold']:.2f}  F1={best['f1']:.3f}  "
              f"sens={best['sensitivity']:.3f}  spec={best['specificity']:.3f}")

        per_fold_rows.append({
            "condition": condition,
            "heldout_slide": slide,
            "n": len(labels),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "positive_class": positive_name,
            # Default-threshold metrics
            "acc_at_0.5": at_default["accuracy"],
            "precision_at_0.5": at_default["precision"],
            "recall_at_0.5": at_default["recall"],
            "f1_at_0.5": at_default["f1"],
            "tp_at_0.5": at_default["tp"],
            "fp_at_0.5": at_default["fp"],
            "fn_at_0.5": at_default["fn"],
            "tn_at_0.5": at_default["tn"],
            # F1-optimal-threshold metrics
            "f1_opt_threshold": best["threshold"],
            "f1_opt": best["f1"],
            "precision_at_f1_opt": best["precision"],
            "recall_at_f1_opt": best["recall"],
            "sensitivity_at_f1_opt": best["sensitivity"],
            "specificity_at_f1_opt": best["specificity"],
            "tp_at_f1_opt": best["tp"],
            "fp_at_f1_opt": best["fp"],
            "fn_at_f1_opt": best["fn"],
            "tn_at_f1_opt": best["tn"],
            # Provenance
            "predictions_csv": str(pf),
            "pr_curve_csv": str(pr_csv),
        })

    # Per-fold table.
    summary_path = root / "scored_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = list(per_fold_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_fold_rows:
            w.writerow(r)

    # Per-condition rollup: mean +/- std across folds for headline numbers.
    by_cond = {}
    for r in per_fold_rows:
        by_cond.setdefault(r["condition"], []).append(r)

    cond_path = root / "scored_per_condition.csv"
    headline_keys = [
        "f1_at_0.5", "precision_at_0.5", "recall_at_0.5", "acc_at_0.5",
        "f1_opt", "precision_at_f1_opt", "recall_at_f1_opt",
        "sensitivity_at_f1_opt", "specificity_at_f1_opt", "f1_opt_threshold",
    ]

    def mean_std(values):
        if not values:
            return 0.0, 0.0
        if len(values) == 1:
            return values[0], 0.0
        return statistics.mean(values), statistics.stdev(values)

    with open(cond_path, "w", newline="") as f:
        fieldnames = ["condition", "n_folds"]
        for k in headline_keys:
            fieldnames += [f"{k}_mean", f"{k}_std"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cond, rows in by_cond.items():
            row = {"condition": cond, "n_folds": len(rows)}
            for k in headline_keys:
                m, s = mean_std([r[k] for r in rows])
                row[f"{k}_mean"] = m
                row[f"{k}_std"] = s
            w.writerow(row)

    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {cond_path}")
    print(f"Per-fold PR curves saved next to each predictions.csv as pr_curve.csv")


if __name__ == "__main__":
    main()
