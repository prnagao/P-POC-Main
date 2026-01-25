# Blind Evaluation (Option B)

This adds a **human-verifiable** evaluation path:
- The model never sees labels.
- You get a per-image CSV of predictions.
- A human adds the `true_label` column.
- A small scorer computes metrics/confusion.

## 1) Drop unlabeled images
Put PNG/JPG/TIF into:
```
blind_eval/images/
```

## 2) Run blind inference
Adapter example:
```
python -m src.pathology_poc.infer_blind   --input_dir blind_eval/images   --output_csv blind_eval/predictions/preds_adapter.csv   --checkpoint models/checkpoints/adapter_neutrophil_rbc.pth   --class_names neutrophil rbc   --img_size 518   --use_adapter
```

This writes a CSV like:
```
filepath,pred_label,prob_neutrophil,prob_rbc,max_prob,model_kind,checkpoint_name,backbone,img_size,timestamp
```

## 3) Human annotates
Open the CSV and add a `true_label` column for each row.

## 4) Score
```
python -m src.pathology_poc.score_blind   --pred_csv blind_eval/predictions/preds_adapter.csv   --class_names neutrophil rbc   --threshold 0.80
```
Outputs metrics to console and `<csv>.metrics.json`.

## Notes
- Default image size is **518** to match DINOv2 ViT‑S/14 common configs.
- If your checkpoint doesn't load strictly, a warning prints; predictions still run with available weights.
- For clinical use, consider keeping only **high-confidence** predictions (e.g., `threshold >= 0.9`) and treat others as `abstain`.
