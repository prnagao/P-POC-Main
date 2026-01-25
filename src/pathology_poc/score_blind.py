import argparse, json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def main():
    ap = argparse.ArgumentParser(description="Score blind predictions after human adds true_label column.")
    ap.add_argument("--pred_csv", required=True, help="CSV from infer_blind.py with a true_label column added.")
    ap.add_argument("--class_names", nargs="+", required=True, help="Class order, e.g., neutrophil rbc.")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Optional: abstain if max_prob < threshold (set >0 to enable)." )
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    if "true_label" not in df.columns:
        raise SystemExit("CSV is missing a 'true_label' column. Please annotate it before scoring.")

    # Only scored rows
    scored = df.dropna(subset=["true_label"]).copy()
    if scored.empty:
        raise SystemExit("No rows with true_label present.")

    cls = args.class_names

    # Optionally implement an abstain policy
    if args.threshold > 0.0:
        # mark low-confidence as 'abstain'
        pred_adj = []
        for _, row in scored.iterrows():
            if float(row.get("max_prob", 1.0)) < args.threshold:
                pred_adj.append("abstain")
            else:
                pred_adj.append(row["pred_label"])
        scored["pred_eval"] = pred_adj
        labels_for_eval = cls + ["abstain"]
    else:
        scored["pred_eval"] = scored["pred_label"]
        labels_for_eval = cls

    y_true = scored["true_label"].tolist()
    y_pred = scored["pred_eval"].tolist()

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_eval)
    report = classification_report(y_true, y_pred, labels=labels_for_eval, target_names=labels_for_eval, output_dict=True, zero_division=0)

    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    print("\nClassification report:")
    for k, v in report.items():
        if isinstance(v, dict) and "precision" in v:
            print(f"{k:>12s}  P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1-score']:.3f}  n={v['support']}")
    acc = report.get("accuracy", 0.0)
    print(f"\nAccuracy: {acc:.3f}")
    out = Path(args.pred_csv).with_suffix(".metrics.json")
    out.write_text(json.dumps({"confusion_matrix": cm.tolist(), "report": report}, indent=2))
    print(f"Wrote metrics JSON: {out}")

if __name__ == "__main__":
    main()
