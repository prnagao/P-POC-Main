from typing import Tuple, Optional
import torch
import torch.nn as nn
import timm

try:
    import torch.hub as _torch_hub
    _HUB_AVAILABLE = True
except Exception:
    _HUB_AVAILABLE = False

class DinoV2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        use_hub: bool = False,
        hub_name: Optional[str] = None,
        use_registers: bool = False,
        img_size: int = 336,
    ):
        super().__init__()
        self.backbone = None
        feat_dim = None

        if use_hub:
            if not _HUB_AVAILABLE:
                raise RuntimeError("PyTorch Hub is not available in this environment.")
            base_hub = hub_name or "dinov2_vits14"
            if use_registers and not base_hub.endswith("_reg"):
                if base_hub in {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}:
                    base_hub = base_hub + "_reg"
            self.backbone = _torch_hub.load('facebookresearch/dinov2', base_hub)
            feat_dim = getattr(self.backbone, "embed_dim", None)
            if feat_dim is None:
                feat_dim = getattr(self.backbone, "num_features", None)
            if feat_dim is None:
                with torch.no_grad():
                    dummy = torch.randn(1, 3, img_size, img_size)
                    out = self.backbone(dummy)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if out.ndim == 2:
                        feat_dim = out.shape[-1]
                    else:
                        feat_dim = out.view(1, -1).shape[-1]
        else:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            feat_dim = getattr(self.backbone, "num_features", None)
            if feat_dim is None:
                feat_dim = getattr(self.backbone, "embed_dim", None)

        if feat_dim is None:
            raise RuntimeError("Unable to determine feature dimension for DINOv2 backbone.")

        self.head = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        logits = self.head(feats)
        return logits

def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
