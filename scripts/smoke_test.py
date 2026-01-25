"""Quick smoke test that uses a dummy batch to verify baseline and adapter forward pass.
Run: python scripts/smoke_test.py
"""
import torch
from src.pathology_poc.models.dinov2 import DinoV2Classifier, count_trainable_params
from src.pathology_poc.models.adapter import AdapterClassifier

IMG_SIZE = 336
BATCH = 2
NUM_CLASSES = 2

def main():
    # Baseline
    base = DinoV2Classifier(num_classes=NUM_CLASSES, freeze_backbone=True, img_size=IMG_SIZE)
    total, trainable = count_trainable_params(base)
    print(f"[baseline] params total={total:,}, trainable={trainable:,}")
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        y = base(x)
    print("[baseline] logits", y.shape)

    # Adapter
    model = AdapterClassifier(num_classes=NUM_CLASSES, freeze_backbone=True, img_size=IMG_SIZE)
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        y = model(x)
    print("[adapter] logits", y.shape)

if __name__ == "__main__":
    main()
