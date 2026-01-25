# DINOv2 Linear-Probe Pathology POC

Minimal, reproducible diagnostic proof-of-concept using frozen DINOv2 features and a linear probe. The goal is clarity and clinician-inspectable outputs, not leaderboard numbers.

## Key principles
- **Tiny dataset on purpose:** ~40–50 training crops, ~10–20 validation crops, ~10–15 hard negatives, plus one fully held-out slide for generalization checks.
- **No augmentation:** transforms are deterministic resize + normalize only. No flips, rotations, color jitter, or stain tricks.
- **Slide-driven splits:** train/val/test are chosen by slide ID, not random sampling. Hard negatives can live on the same slide now and be swapped for a dedicated slide later without refactoring.
- **Frozen backbone:** ViT-S/14 DINOv2 stays frozen; only the linear head trains.
- **Human-in-the-loop outputs:** every inference/export writes a CSV with filename, slide ID, true label, predicted probability (positive class), and predicted class for microscopy review.

## Data layout
```
dataset/
  slide_S1/
    positive/*.png
  slide_S2/
    positive/*.png
  slide_S3/
    negative/*.png
  slide_S4/
    negative/*.png
  hard_negative_slide_X/
    negative/*.png
dataset_unused/
```
Class folder names come from `data.class_names` in the config (first entry is treated as the positive class unless overridden).

## Configuration
`config/default.yaml` controls slide splits, class labels, hyperparameters, and output paths. Example defaults:
```
project_name: dinov2_slide_poc

data:
  root: dataset
  img_size: 518
  class_names: ["positive", "negative"]
  positive_class: "positive"
  splits:
    train: ["train_slide_A", "train_slide_B"]
    val:   ["val_slide_A"]
    test:  ["test_slide_A"]
    hard_negative: []

train:
  batch_size: 4
  epochs: 5
  lr: 0.0001
  weight_decay: 0.01
  num_workers: 2
  seed: 42

model:
  backbone: vit_small_patch14_dinov2.lvd142m  # frozen

output:
  base_dir: outputs
  checkpoint_dir: outputs/checkpoints
  inference_csv: outputs/predictions.csv
```
project_name: dinov2_slide_poc

data:
  root: dataset
  img_size: 518
  class_names: ["positive", "negative"]
  positive_class: "positive"
  splits:
    train: ["slide_S1","slide_S3"]
    test:   ["slide_S2","slide_S4"]    
    hard_negative: ["hard_negative_slide_X"]  # optional, can be empty on disk

## Single entry-point CLI
Use `python -m src.pathology_poc.cli` with one of three subcommands. All commands respect the YAML config; no random splits are created in code.

### Train (linear probe only)
```
python -m src.pathology_poc.cli train --config config/default.yaml
```
- Builds loaders from slide IDs
- Freezes DINOv2 backbone
- Tracks best validation accuracy and saves checkpoints under `output.checkpoint_dir`

### Evaluate on a held-out split
```
python -m src.pathology_poc.cli eval \
  --config config/default.yaml \
  --checkpoint outputs/checkpoints/<ckpt>.pth \
  --split test  # or hard_negative
```
- Runs deterministic transforms only
- Reports precision/recall/F1 (aggregate secondary)
- Writes per-image CSV for manual review (`output.inference_csv` or `--output`)

### Inference / human review export
```
python -m src.pathology_poc.cli infer \
  --config config/default.yaml \
  --checkpoint outputs/checkpoints/<ckpt>.pth \
  --split hard_negative  # or test/val/train
```
Produces the CSV required for microscope review with the columns: filename, slide_id, true_label, predicted_probability, predicted_class.

## Philosophy & future work
- This repo mirrors earlier RBC vs Neutrophil and Plasmodium POCs but keeps engineering minimal for interpretability.
- No augmentation is intentional; add stain/rotation/color pipelines later if you need robustness.
- Future iterations can plug in larger datasets, true separate hard-negative slides, and richer metrics without restructuring the code paths here.
