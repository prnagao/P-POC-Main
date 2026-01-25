#!/usr/bin/env python3
"""
infer_fov.py

Label-blind 100× FOV inference pipeline:
checkpoint -> FOV images -> deterministic tiling -> per-tile probabilities -> per-FOV aggregation

Outputs:
  - per_tile.csv  (REQUIRED)
  - per_fov.csv   (REQUIRED)
  - top_tiles.csv (optional; coordinate list of top-probability tiles)
  - overlays/*.png (optional; simple rectangle overlays)

Critical safeguard (reviewer-critical):
  - The model NEVER receives paths/filenames/folder names/labels. Only image tensors.
  - Labels (CSV or folder-based) are joined strictly post hoc AFTER predictions are written.

Dependencies: torch, torchvision, PIL, pandas, numpy
CPU-compatible; uses torch.no_grad() and model.eval()
"""

import argparse
import ast
import importlib
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn


# -------------------------
# Label-blind safeguard (enforced by design)
# -------------------------
# This script NEVER passes any file/folder names, paths, or labels into the model.
# The model only receives image tensors.
# Any label parsing (CSV or folder-based) happens strictly AFTER predictions are written to disk.


# -------------------------
# Import canonical eval transform (preferred)
# -------------------------
def _try_import_eval_transform():
    """
    Prefer canonical transform from:
      src/pathology_poc/data/datasets.py

    Falls back to an equivalent local implementation ONLY if import fails.
    """
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent
    if (repo_root / "src").exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from src.pathology_poc.data.datasets import make_eval_transform, VALID_EXTENSIONS  # type: ignore

        return make_eval_transform, VALID_EXTENSIONS
    except Exception:
        # Fallback: identical to canonical datasets.py (eval path only)
        from torchvision import transforms  # type: ignore

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        def make_eval_transform(img_size: int) -> transforms.Compose:
            return transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

        return make_eval_transform, VALID_EXTENSIONS


MAKE_EVAL_TRANSFORM, VALID_EXTENSIONS = _try_import_eval_transform()


# -------------------------
# Minimal YAML loader (dependency-light)
# -------------------------
def _parse_scalar(val: str) -> Any:
    val = val.strip()
    if val == "" or val.lower() == "null":
        return None
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    if re.match(r"^[+-]?\d+$", val):
        try:
            return int(val)
        except Exception:
            pass
    if re.match(r"^[+-]?\d*\.\d+$", val):
        try:
            return float(val)
        except Exception:
            pass
    if (val.startswith("[") and val.endswith("]")) or (val.startswith("{") and val.endswith("}")):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads YAML config.

    - Tries PyYAML if installed.
    - Falls back to a minimal YAML subset parser (sufficient for the provided config/default.yaml style).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Try PyYAML if available (optional dependency)
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config YAML did not parse to a dict.")
        return cfg
    except Exception:
        pass

    # Minimal parser (supports simple nested dicts + lists)
    text = path.read_text(encoding="utf-8").splitlines()

    processed: List[str] = []
    for line in text:
        raw = line.rstrip("\n")
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            continue
        # Remove inline comments that begin after whitespace
        if "#" in raw:
            m = re.search(r"\s#", raw)
            if m:
                raw = raw[: m.start()].rstrip()
        if raw.strip():
            processed.append(raw)

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]

    for raw in processed:
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.lstrip(" ")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        # list item
        if line.startswith("- "):
            item = _parse_scalar(line[2:])
            if not isinstance(parent, list):
                raise ValueError(f"YAML parse error: list item under non-list: {raw}")
            parent.append(item)
            continue

        if ":" not in line:
            raise ValueError(f"YAML parse error: missing ':' in line: {raw}")

        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        if rest == "":
            # Decide whether this should be a list or dict container.
            # We'll default to dict; lists are created when we see a subsequent '- ' under it.
            new_container: Dict[str, Any] = {}
            if not isinstance(parent, dict):
                raise ValueError(f"YAML parse error: mapping under non-dict: {raw}")
            parent[key] = new_container
            stack.append((indent, new_container))
        else:
            val = _parse_scalar(rest)
            if not isinstance(parent, dict):
                raise ValueError(f"YAML parse error: key/value under non-dict: {raw}")
            parent[key] = val

        # Convert empty dict to list if next processed lines indicate list items
        # (handled implicitly by raising if '- ' appears under a dict; users should prefer PyYAML)


    return root


# -------------------------
# Tiling (last-tile-anchored strategy)
# -------------------------
def compute_start_positions(image_size: int, tile_size: int, stride: int) -> List[int]:
    """
    Start positions: 0, stride, 2*stride, ...
    If final tile doesn't align with image edge, include tile starting at (image_size - tile_size).
    """
    if image_size <= tile_size:
        return [0]
    positions = list(range(0, image_size - tile_size + 1, stride))
    last = image_size - tile_size
    if positions and positions[-1] != last:
        positions.append(last)
    if not positions:
        positions = [last]
    return positions


def iter_tiles(img: Image.Image, tile_size: int, stride: int) -> Iterable[Tuple[int, int, Image.Image]]:
    w, h = img.size
    xs = compute_start_positions(w, tile_size, stride)
    ys = compute_start_positions(h, tile_size, stride)
    for y in ys:
        for x in xs:
            yield x, y, img.crop((x, y, x + tile_size, y + tile_size))


# -------------------------
# Model: DINOv2 + (optional) adapter + head (fallback architecture)
# -------------------------
class MLPAdapter(nn.Module):
    """Residual MLP adapter on feature vectors (B, D) -> delta (B, D)."""

    def __init__(self, dim: int, hidden_ratio: float = 0.25, dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        hidden_dim = max(1, int(dim * hidden_ratio))
        self.ln = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.down(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.up(y)
        y = self.drop(y)
        return y


class DinoClassifier(nn.Module):
    """backbone(image)->features -> optional adapter -> linear head -> logits"""

    def __init__(self, backbone: nn.Module, feat_dim: int, num_outputs: int, adapter: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = nn.Linear(feat_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)

        # DINOv2 hub models often return dicts
        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats:
                feats = feats["x_norm_clstoken"]
            elif "cls_token" in feats:
                feats = feats["cls_token"]
            else:
                # first tensor value fallback
                for v in feats.values():
                    if torch.is_tensor(v):
                        feats = v
                        break

        # If (B, T, D), take CLS token
        if torch.is_tensor(feats) and feats.ndim == 3:
            feats = feats[:, 0, :]

        if not torch.is_tensor(feats) or feats.ndim != 2:
            raise RuntimeError(f"Backbone produced unexpected features: type={type(feats)}, ndim={getattr(feats, 'ndim', None)}")

        if self.adapter is not None:
            feats = feats + self.adapter(feats)

        return self.head(feats)


def _infer_backbone_feat_dim(backbone: nn.Module, img_size: int) -> int:
    for attr in ("embed_dim", "num_features", "feature_dim", "dim"):
        if hasattr(backbone, attr):
            val = getattr(backbone, attr)
            if isinstance(val, int) and val > 0:
                return val

    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size)
        out = backbone(x)

        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                out = out["x_norm_clstoken"]
            else:
                for v in out.values():
                    if torch.is_tensor(v):
                        out = v
                        break

        if torch.is_tensor(out) and out.ndim == 3:
            out = out[:, 0, :]

        if not torch.is_tensor(out) or out.ndim != 2:
            raise RuntimeError("Could not infer backbone feature dim from dummy forward.")
        return int(out.shape[1])


def build_backbone_from_config(cfg: Dict[str, Any], hub_repo: str, hub_dir: Optional[str]) -> nn.Module:
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    hub_name = model_cfg.get("hub_name", "dinov2_vits14")

    if hub_dir is not None:
        repo_or_dir = hub_dir
        source = "local"
    else:
        repo_or_dir = hub_repo
        source = "github"

    try:
        return torch.hub.load(repo_or_dir, hub_name, source=source, pretrained=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv2 backbone via torch.hub ({repo_or_dir}, model={hub_name}, source={source}). "
            f"Error: {e}\n"
            f"Tips:\n"
            f"  - If offline, pass --hub_dir /path/to/local/clone/of/facebookresearch/dinov2\n"
            f"  - Ensure weights are cached/available."
        )


# -------------------------
# Checkpoint loading (robust, with safety checks)
# -------------------------
def _normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def strip_prefix(k: str, prefix: str) -> str:
        return k[len(prefix) :] if k.startswith(prefix) else k

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        k2 = k
        for p in ("module.", "model.", "net."):
            k2 = strip_prefix(k2, p)
        out[k2] = v
    return out


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Extract a tensor state_dict from different checkpoint formats.

    Supports checkpoints that store weights under:
      - ckpt_obj["model_state"] (your adapter checkpoint format)
      - ckpt_obj["state_dict"], ckpt_obj["model_state_dict"]
      - or a raw state_dict dict-of-tensors.
    """
    if isinstance(ckpt_obj, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                sd = ckpt_obj[key]
                if all(torch.is_tensor(v) for v in sd.values()):
                    return sd

        if all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj  # raw state_dict

    raise ValueError(
        "Could not extract a tensor state_dict from checkpoint. "
        "Expected a dict of tensors under one of: model_state, model_state_dict, state_dict."
    )


def _guess_head_weight_key(sd: Dict[str, torch.Tensor], num_classes: int) -> Tuple[str, Optional[str], int]:
    """
    Returns (weight_key, bias_key_or_None, num_outputs)

    Supports:
      - multi-class head: weight shape (num_classes, D)
      - binary head:     weight shape (1, D)
    """
    candidates: List[Tuple[str, torch.Tensor]] = []
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.ndim != 2:
            continue
        if v.shape[0] in (num_classes, 1) and v.shape[1] >= 32:
            candidates.append((k, v))

    if not candidates:
        raise RuntimeError(
            f"Could not find head weight with shape (num_classes={num_classes} or 1, feat_dim>=32) in checkpoint."
        )

    # Prefer the candidate that matches num_classes; otherwise accept binary
    # Within each group, prefer larger feature dim.
    def _rank(item: Tuple[str, torch.Tensor]) -> Tuple[int, int]:
        k, v = item
        is_multiclass = 1 if int(v.shape[0]) == int(num_classes) else 0
        return (is_multiclass, int(v.shape[1]))

    candidates.sort(key=_rank, reverse=True)
    w_key, w = candidates[0]
    num_outputs = int(w.shape[0])

    b_key = None
    if w_key.endswith(".weight"):
        maybe_b = w_key[: -len(".weight")] + ".bias"
        if (
            maybe_b in sd
            and torch.is_tensor(sd[maybe_b])
            and sd[maybe_b].ndim == 1
            and int(sd[maybe_b].shape[0]) == num_outputs
        ):
            b_key = maybe_b

    return w_key, b_key, num_outputs


def _try_build_model_from_repo(
    cfg: Dict[str, Any],
    mode: str,
    num_classes: int,
    device: torch.device,
    explicit_builder: Optional[str] = None,
) -> Optional[nn.Module]:
    """
    Best-effort: if your repo defines the exact adapter/linear architectures, this path loads checkpoints best.

    If explicit_builder is provided, it must be "module.submodule:function_name".

    Otherwise, we search common builder locations and also scan src.pathology_poc.* for
    build_model/get_model functions.
    """

    def _call_builder(fn: Any) -> Optional[nn.Module]:
        if not callable(fn):
            return None
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        kwargs: Dict[str, Any] = {}
        if sig is not None:
            for pname in sig.parameters.keys():
                if pname in ("cfg", "config"):
                    kwargs[pname] = cfg
                elif pname in ("mode", "model_mode"):
                    kwargs[pname] = mode
                elif pname in ("num_classes", "n_classes", "classes"):
                    kwargs[pname] = num_classes

        try:
            model = fn(**kwargs) if kwargs else fn()
            if isinstance(model, nn.Module):
                model.to(device)
                model.eval()
                return model
        except Exception:
            return None
        return None

    # Explicit builder path
    if explicit_builder:
        if ":" in explicit_builder:
            # accept module:function too
            mod_name, fn_name = explicit_builder.split(":", 1)
        elif "." in explicit_builder:
            parts = explicit_builder.split(".")
            mod_name, fn_name = ".".join(parts[:-1]), parts[-1]
        else:
            mod_name, fn_name = explicit_builder, "build_model"
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            model = _call_builder(fn)
            if model is not None:
                return model
        except Exception:
            return None

    # Common candidates
    candidates = [
        ("src.pathology_poc.models", "build_model"),
        ("src.pathology_poc.models", "get_model"),
        ("src.pathology_poc.model", "build_model"),
        ("src.pathology_poc.model", "get_model"),
        ("src.pathology_poc.modeling", "build_model"),
        ("src.pathology_poc.modeling", "get_model"),
        ("src.pathology_poc.model_builder", "build_model"),
        ("src.pathology_poc.model_builder", "get_model"),
    ]

    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            model = _call_builder(fn)
            if model is not None:
                return model
        except Exception:
            continue

    # Optional: scan package for any module exposing build_model/get_model.
    # This is a best-effort convenience and is safe (no inference yet).
    try:
        import pkgutil

        pkg_name = "src.pathology_poc"
        pkg = importlib.import_module(pkg_name)
        prefix = pkg.__name__ + "."
        for modinfo in pkgutil.walk_packages(pkg.__path__, prefix):
            try:
                mod = importlib.import_module(modinfo.name)
            except Exception:
                continue
            for fn_name in ("build_model", "get_model"):
                fn = getattr(mod, fn_name, None)
                model = _call_builder(fn)
                if model is not None:
                    return model
    except Exception:
        pass

    return None


def _try_load_global_adapter_by_shape(adapter: Optional[nn.Module], sd: Dict[str, torch.Tensor]) -> bool:
    """
    Conservative shape-based adapter loader for the fallback MLPAdapter.
    Returns True if uniquely identified and loaded, else False.
    """
    if adapter is None:
        return True
    if not isinstance(adapter, MLPAdapter):
        return False

    dim = int(adapter.down.in_features)
    hidden = int(adapter.down.out_features)

    def find_unique(shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
        matches = [v for v in sd.values() if torch.is_tensor(v) and tuple(v.shape) == shape]
        if len(matches) == 1:
            return matches[0]
        return None

    down_w = find_unique((hidden, dim))
    up_w = find_unique((dim, hidden))
    if down_w is None or up_w is None:
        return False

    mapped: Dict[str, torch.Tensor] = {"down.weight": down_w, "up.weight": up_w}
    down_b = find_unique((hidden,))
    up_b = find_unique((dim,))
    if down_b is not None:
        mapped["down.bias"] = down_b
    if up_b is not None:
        mapped["up.bias"] = up_b

    if isinstance(adapter.ln, nn.LayerNorm):
        # LayerNorm weights are ambiguous; only load if unique (rare)
        ln_w = find_unique((dim,))
        ln_b = find_unique((dim,))
        if ln_w is not None and ln_b is not None:
            mapped["ln.weight"] = ln_w
            mapped["ln.bias"] = ln_b

    adapter.load_state_dict(mapped, strict=False)
    return True


def load_model_for_inference(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    mode: str,
    device: torch.device,
    hub_repo: str,
    hub_dir: Optional[str],
    model_builder: Optional[str] = None,
) -> Tuple[nn.Module, int]:
    """
    Returns:
      model (eval-ready)
      positive_class_index (for softmax) OR 0 for binary sigmoid head.

    Safety rules:
      - Adapter mode refuses to run unless adapter weights are loaded (or a repo-native model loads cleanly).
      - Classification head weights are verified to load from checkpoint.
    """
    data_cfg = cfg.get("data", {})
    class_names = data_cfg.get("class_names", ["positive", "negative"])
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("cfg['data']['class_names'] must be a list with at least 2 entries.")

    positive_name = data_cfg.get("positive_class", class_names[0])
    if positive_name not in class_names:
        raise ValueError(f"positive_class={positive_name} not found in class_names={class_names}")

    pos_idx = int(class_names.index(positive_name))
    num_classes = int(len(class_names))
    img_size = int(data_cfg.get("img_size", 518))

    ckpt_obj = torch.load(checkpoint_path, map_location="cpu")

    # Full pickled model shortcut
    if isinstance(ckpt_obj, nn.Module):
        model = ckpt_obj
        model.eval()
        model.to(device)
        return model, pos_idx

    # Some checkpoints store a pickled module under a key
    if isinstance(ckpt_obj, dict):
        for key in ("model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], nn.Module):
                model = ckpt_obj[key]
                model.eval()
                model.to(device)
                return model, pos_idx

    sd_raw = _extract_state_dict(ckpt_obj)
    sd = _normalize_state_dict_keys(sd_raw)

    # Determine whether checkpoint is adapter/linear. Prefer checkpoint metadata if present.
    ckpt_kind = None
    if isinstance(ckpt_obj, dict):
        ckpt_kind = str(ckpt_obj.get("model_kind", "")).lower()

    ckpt_has_adapter = ("adapter" in ckpt_kind) if ckpt_kind else any("adapter" in k.lower() for k in sd.keys())
    if mode == "adapter" and not ckpt_has_adapter:
        raise RuntimeError(
            f"Requested mode=adapter, but this checkpoint does not look like an adapter checkpoint "
            f"(model_kind={ckpt_kind!r}, no 'adapter' keys found). Use --mode linear or provide an adapter checkpoint."
        )

    # Find head tensor in checkpoint and capture expected output dim.
    head_w_key, head_b_key, head_out = _guess_head_weight_key(sd, num_classes=num_classes)
    head_w = sd[head_w_key].detach().cpu()

    # If checkpoint appears to be binary head, use sigmoid semantics.
    # For binary head, positive_index is unused; we return 0.
    pos_idx_effective = 0 if head_out == 1 else pos_idx

    # Prefer repo-native model builder if available (best for per-layer adapters)
    model_from_repo = _try_build_model_from_repo(
        cfg=cfg, mode=mode, num_classes=num_classes, device=device, explicit_builder=model_builder
    )
    if model_from_repo is not None:
        model = model_from_repo
        model.load_state_dict(sd, strict=False)

        # Verify head loaded: find an exact matching tensor by value (allclose)
        found = False
        for _, v in model.state_dict().items():
            if torch.is_tensor(v) and tuple(v.shape) == tuple(head_w.shape) and torch.allclose(v.detach().cpu(), head_w):
                found = True
                break
        if not found:
            raise RuntimeError(
                "Repo model builder was found, but checkpoint head weights do not appear to have loaded. "
                "Refusing to run inference with a likely mismatch."
            )

        model.eval()
        model.to(device)
        return model, pos_idx_effective

    # Fallback minimal wrapper (only safe for global feature-space adapters)
    backbone = build_backbone_from_config(cfg, hub_repo=hub_repo, hub_dir=hub_dir)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    feat_dim = _infer_backbone_feat_dim(backbone, img_size=img_size)

    adapter: Optional[nn.Module] = None
    if mode == "adapter":
        mcfg = cfg.get("model", {})
        adapter = MLPAdapter(
            dim=feat_dim,
            hidden_ratio=float(mcfg.get("adapter_hidden_ratio", 0.25)),
            dropout=float(mcfg.get("adapter_dropout", 0.1)),
            use_layernorm=bool(mcfg.get("adapter_use_layernorm", True)),
        )

    model = DinoClassifier(backbone=backbone, feat_dim=feat_dim, num_outputs=head_out, adapter=adapter)

    # Load what we can (strict=False so backbone key mismatches don't matter)
    model.load_state_dict(sd, strict=False)

    # Map head by shape if not loaded by name
    matched_keys = set(sd.keys()) & set(model.state_dict().keys())
    head_matched = any(k.startswith("head.") for k in matched_keys)
    if not head_matched:
        mapped = {"head.weight": sd[head_w_key]}
        if head_b_key is not None:
            mapped["head.bias"] = sd[head_b_key]
        model.load_state_dict(mapped, strict=False)

    # Verify head loaded
    if not torch.allclose(model.head.weight.detach().cpu(), head_w):
        raise RuntimeError("Failed to load classification head weights from checkpoint.")

    # Adapter loading: must load (adapter mode)
    if adapter is not None:
        adapter_sd = {k: v for k, v in sd.items() if k.startswith("adapter.")}
        loaded_ok = False

        if adapter_sd:
            model.load_state_dict(adapter_sd, strict=False)
            # Verify at least one adapter tensor matches
            for k, v in adapter_sd.items():
                if k in model.state_dict() and torch.allclose(model.state_dict()[k].detach().cpu(), v.detach().cpu()):
                    loaded_ok = True
                    break
        else:
            loaded_ok = _try_load_global_adapter_by_shape(model.adapter, sd)

        if not loaded_ok:
            raise RuntimeError(
                "mode=adapter but adapter weights could not be loaded confidently from the checkpoint.\n"
                "Refusing to run inference with a random/incorrect adapter.\n"
                "If your adapter is implemented inside transformer blocks (per-layer), ensure the repo's native "
                "model builder is importable (or pass --model_builder), or extend mapping logic for that architecture."
            )

    model.eval()
    model.to(device)
    return model, pos_idx_effective


# -------------------------
# Probabilities + aggregation
# -------------------------
def logits_to_positive_prob(logits: torch.Tensor, positive_index: int) -> torch.Tensor:
    """
    Returns: (B,) tensor in [0,1]
    Supports:
      - binary sigmoid head: logits shape (B,1)
      - multi-class softmax head: logits shape (B,C)
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (B,C). Got: {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    probs = torch.softmax(logits, dim=1)
    return probs[:, int(positive_index)]


def aggregate_fov_scores(
    tile_probs: Sequence[float],
    top_ks: Sequence[int] = (3, 5),
    threshold_T: float = 0.5,
) -> Dict[str, Any]:
    probs = np.asarray(tile_probs, dtype=np.float32)
    if probs.size == 0:
        raise ValueError("No tile probabilities provided for aggregation.")

    max_score = float(np.max(probs))
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    sorted_probs = np.sort(probs)[::-1]

    out: Dict[str, Any] = {
        "FOV_score_max": max_score,
        "num_tiles": int(probs.size),
        "num_tiles_above_T": int(np.sum(probs >= float(threshold_T))),
        "top_tile_index": top_idx,
        "top_tile_probability": top_prob,
    }
    for k in top_ks:
        kk = int(k)
        k_eff = min(kk, int(probs.size))
        out[f"FOV_score_top{kk}"] = float(np.mean(sorted_probs[:k_eff]))
    return out


# -------------------------
# Labels (optional, post-hoc only)
# -------------------------
def load_labels_csv(labels_csv: str) -> Dict[str, Any]:
    df = pd.read_csv(labels_csv)
    if "fov_id" not in df.columns or "true_label" not in df.columns:
        raise ValueError("labels_csv must have columns: fov_id,true_label")
    return {str(r["fov_id"]): r["true_label"] for _, r in df.iterrows()}


def infer_labels_from_folders(
    fov_paths: Sequence[Path],
    class_names: Sequence[str],
    max_depth_up: int = 4,
) -> Dict[str, str]:
    """
    Folder-based labels: positive/ / negative/

    IMPORTANT: called ONLY after inference outputs are written to disk.
    """
    class_set = {c.lower() for c in class_names}
    out: Dict[str, str] = {}
    for p in fov_paths:
        label = None
        cur = p.parent
        for _ in range(max_depth_up):
            name = cur.name.lower()
            if name in class_set:
                label = name
                break
            if cur.parent == cur:
                break
            cur = cur.parent
        if label is not None:
            out[p.name] = label  # filename only
    return out


def normalize_true_label(val: Any, positive_class_name: str) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None

    if isinstance(val, (int, np.integer)):
        return 1 if int(val) != 0 else 0
    if isinstance(val, (float, np.floating)):
        return 1 if float(val) >= 0.5 else 0

    s = str(val).strip().lower()
    if s == str(positive_class_name).strip().lower():
        return 1
    return 0


# -------------------------
# Threshold sweep (false-negative-first)
# -------------------------
@dataclass
class SweepResult:
    method: str
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float
    specificity: float
    fp_per_negative: float


def _confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, tn, fn


def sweep_thresholds_for_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    method_name: str,
    thresholds: np.ndarray,
) -> List[SweepResult]:
    results: List[SweepResult] = []
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_neg == 0 or n_pos == 0:
        return results

    for t in thresholds:
        y_pred = (scores >= t).astype(np.int32)
        tp, fp, tn, fn = _confusion_from_preds(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        fp_per_neg = fp / n_neg if n_neg > 0 else float("nan")
        results.append(
            SweepResult(
                method=str(method_name),
                threshold=float(t),
                tp=int(tp),
                fp=int(fp),
                tn=int(tn),
                fn=int(fn),
                sensitivity=float(sens),
                specificity=float(spec),
                fp_per_negative=float(fp_per_neg),
            )
        )
    return results


def sweep_thresholds_for_multihit(
    y_true: np.ndarray,
    tile_probs_list: Sequence[np.ndarray],
    N: int,
    method_name: str,
    thresholds: np.ndarray,
) -> List[SweepResult]:
    results: List[SweepResult] = []
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_neg == 0 or n_pos == 0:
        return results

    Nn = int(N)
    for t in thresholds:
        preds: List[int] = []
        for probs in tile_probs_list:
            preds.append(int(np.sum(probs >= t) >= Nn))
        y_pred = np.asarray(preds, dtype=np.int32)
        tp, fp, tn, fn = _confusion_from_preds(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        fp_per_neg = fp / n_neg if n_neg > 0 else float("nan")
        results.append(
            SweepResult(
                method=str(method_name),
                threshold=float(t),
                tp=int(tp),
                fp=int(fp),
                tn=int(tn),
                fn=int(fn),
                sensitivity=float(sens),
                specificity=float(spec),
                fp_per_negative=float(fp_per_neg),
            )
        )
    return results


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Label-blind 100× FOV inference (tiling + aggregation).")

    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing FOV images (searched recursively).")

    ap.add_argument("--config", type=str, required=True, help="Path to config/default.yaml")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    ap.add_argument("--mode", type=str, default=None, choices=["linear", "adapter"], help="Override model mode.")
    ap.add_argument("--model_builder", type=str, default=None, help="Optional explicit builder 'module:function' (helps adapter loading).")

    ap.add_argument("--tile_size", type=int, default=518)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16, help="Tile batch size for inference.")

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--threshold_T", type=float, default=0.5, help="Threshold used for num_tiles_above_T + quick multi-hit.")
    ap.add_argument("--topk", type=int, nargs="*", default=[3, 5], help="Top-K values for top-k mean aggregation.")
    ap.add_argument("--multi_hit_N", type=int, nargs="*", default=[1, 2], help="N values for multi-hit rule.")

    ap.add_argument("--save_overlays", action="store_true", help="Save rectangle overlays for top tiles.")
    ap.add_argument("--overlay_top_n", type=int, default=10)

    ap.add_argument("--save_top_tiles_csv", action="store_true", help="Write top_tiles.csv listing top tiles per FOV.")
    ap.add_argument("--top_tiles_csv_n", type=int, default=10)

    ap.add_argument("--labels_csv", type=str, default=None, help="Optional CSV with columns: fov_id,true_label (post-hoc eval).")

    ap.add_argument(
        "--labels_from_folders",
        action="store_true",
        help="Infer labels from parent folders (positive/negative) AFTER inference outputs are written.",
    )

    ap.add_argument("--sweep", action="store_true", help="If labels provided, sweep thresholds and write metrics CSV.")
    ap.add_argument("--target_sensitivity", type=float, default=0.99)

    ap.add_argument("--hub_repo", type=str, default="facebookresearch/dinov2")
    ap.add_argument("--hub_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()
    cfg = load_config(args.config)

    cfg_mode = str(cfg.get("model", {}).get("mode", "linear"))
    mode = args.mode if args.mode is not None else cfg_mode
    if mode not in ("linear", "adapter"):
        raise ValueError(f"Invalid mode={mode}. Expected 'linear' or 'adapter'.")

    device = torch.device(args.device)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")

    # Find all images recursively, but keep IDs label-blind (filename only)
    fov_paths = sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS],
        key=lambda p: p.name,
    )
    if len(fov_paths) == 0:
        raise RuntimeError(f"No images found under {args.input_dir} with extensions={sorted(VALID_EXTENSIONS)}")

    # Enforce uniqueness of filename-only IDs (otherwise evaluation/outputs become ambiguous)
    names = [p.name for p in fov_paths]
    if len(names) != len(set(names)):
        from collections import Counter

        dup = [n for n, c in Counter(names).items() if c > 1]
        raise RuntimeError(
            "Duplicate filenames detected under input_dir. Because fov_id is filename-only, IDs must be unique.\n"
            f"Duplicates (first 20): {dup[:20]}\n"
            "Fix: rename files to be unique (recommended) or avoid mixing duplicates in a single run."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    model, pos_idx = load_model_for_inference(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        mode=mode,
        device=device,
        hub_repo=args.hub_repo,
        hub_dir=args.hub_dir,
        model_builder=args.model_builder,
    )
    model.eval()

    img_size = int(cfg.get("data", {}).get("img_size", 518))
    if int(args.tile_size) != int(img_size):
        print(f"[WARN] tile_size={args.tile_size} != cfg.data.img_size={img_size}. Tiles will be resized to {img_size} for the model.")

    transform = MAKE_EVAL_TRANSFORM(img_size)

    tile_rows: List[Dict[str, Any]] = []
    fov_rows: List[Dict[str, Any]] = []
    top_tiles_rows: List[Dict[str, Any]] = []

    for fov_i, fov_path in enumerate(fov_paths):
        fov_id = fov_path.name  # filename only (no directory info)
        img = Image.open(fov_path).convert("RGB")
        w, h = img.size
        if (w, h) != (2448, 2048):
            print(f"[WARN] {fov_id}: expected 2448x2048, got {w}x{h}. Proceeding.")

        coords: List[Tuple[int, int]] = []
        probs: List[float] = []

        # Stream tiles in batches to avoid holding all tensors in memory
        batch_tiles: List[torch.Tensor] = []
        batch_coords: List[Tuple[int, int]] = []

        with torch.no_grad():
            for x, y, tile in iter_tiles(img, tile_size=int(args.tile_size), stride=int(args.stride)):
                # label-blind: we DO NOT pass x,y or filenames into the model; only tensors
                t = transform(tile)
                batch_tiles.append(t)
                batch_coords.append((int(x), int(y)))

                if len(batch_tiles) >= int(args.batch_size):
                    batch = torch.stack(batch_tiles).to(device)
                    logits = model(batch)
                    pos_probs = logits_to_positive_prob(logits, positive_index=pos_idx)
                    pos_probs_np = pos_probs.detach().cpu().numpy().astype(np.float32)

                    for (cx, cy), p in zip(batch_coords, pos_probs_np):
                        coords.append((cx, cy))
                        probs.append(float(p))

                    batch_tiles.clear()
                    batch_coords.clear()

            # Flush last partial batch
            if batch_tiles:
                batch = torch.stack(batch_tiles).to(device)
                logits = model(batch)
                pos_probs = logits_to_positive_prob(logits, positive_index=pos_idx)
                pos_probs_np = pos_probs.detach().cpu().numpy().astype(np.float32)

                for (cx, cy), p in zip(batch_coords, pos_probs_np):
                    coords.append((cx, cy))
                    probs.append(float(p))

                batch_tiles.clear()
                batch_coords.clear()

        if len(probs) == 0:
            print(f"[WARN] {fov_id}: produced 0 tiles; skipping.")
            continue

        # Per-tile rows
        for idx, ((x, y), p) in enumerate(zip(coords, probs)):
            tile_rows.append(
                {
                    "fov_id": fov_id,
                    "tile_index": int(idx),
                    "x": int(x),
                    "y": int(y),
                    "predicted_probability": float(p),
                }
            )

        # Per-FOV aggregation
        agg = aggregate_fov_scores(tile_probs=probs, top_ks=args.topk, threshold_T=args.threshold_T)
        top_tile_index = int(agg["top_tile_index"])
        top_x, top_y = coords[top_tile_index]

        fov_row: Dict[str, Any] = {
            "fov_id": fov_id,
            "FOV_score_max": float(agg["FOV_score_max"]),
            **{k: float(agg[k]) for k in agg.keys() if k.startswith("FOV_score_top")},
            "num_tiles": int(agg["num_tiles"]),
            "num_tiles_above_T": int(agg["num_tiles_above_T"]),
            "top_tile_x": int(top_x),
            "top_tile_y": int(top_y),
            "top_tile_probability": float(agg["top_tile_probability"]),
        }

        num_above_T = int(agg["num_tiles_above_T"])
        for N in args.multi_hit_N:
            Nn = int(N)
            fov_row[f"is_positive_multihit_N{Nn}_T{float(args.threshold_T):g}"] = bool(num_above_T >= Nn)

        fov_rows.append(fov_row)

        # Optional: top tiles list + overlays
        if args.save_top_tiles_csv or args.save_overlays:
            probs_np = np.asarray(probs, dtype=np.float32)
            top_n = min(int(max(args.top_tiles_csv_n, args.overlay_top_n)), int(probs_np.size))
            order = np.argsort(-probs_np)[:top_n]  # descending

            if args.save_top_tiles_csv:
                for rank, idx in enumerate(order[: int(args.top_tiles_csv_n)], start=1):
                    x, y = coords[int(idx)]
                    top_tiles_rows.append(
                        {
                            "fov_id": fov_id,
                            "rank": int(rank),
                            "tile_index": int(idx),
                            "x": int(x),
                            "y": int(y),
                            "predicted_probability": float(probs_np[int(idx)]),
                        }
                    )

            if args.save_overlays:
                overlay_img = img.convert("RGBA")
                draw = ImageDraw.Draw(overlay_img, "RGBA")
                for idx in order[: int(args.overlay_top_n)]:
                    x, y = coords[int(idx)]
                    draw.rectangle(
                        [x, y, x + int(args.tile_size) - 1, y + int(args.tile_size) - 1],
                        outline=(255, 0, 0, 220),
                        width=3,
                    )
                overlay_path = overlays_dir / f"{Path(fov_id).stem}_overlay.png"
                overlay_img.save(overlay_path)

        if (fov_i + 1) % 10 == 0 or (fov_i + 1) == len(fov_paths):
            print(f"[INFO] Processed {fov_i+1}/{len(fov_paths)} FOVs")

    # Write inference outputs
    tiles_csv = out_dir / "per_tile.csv"
    fovs_csv = out_dir / "per_fov.csv"
    pd.DataFrame(tile_rows).to_csv(tiles_csv, index=False)
    pd.DataFrame(fov_rows).to_csv(fovs_csv, index=False)
    print(f"[OK] Wrote per-tile CSV: {tiles_csv}")
    print(f"[OK] Wrote per-FOV  CSV: {fovs_csv}")

    if args.save_top_tiles_csv and top_tiles_rows:
        top_tiles_csv = out_dir / "top_tiles.csv"
        pd.DataFrame(top_tiles_rows).to_csv(top_tiles_csv, index=False)
        print(f"[OK] Wrote top-tiles CSV: {top_tiles_csv}")

    # -------------------------
    # Post-hoc evaluation (ONLY if labels provided)
    # -------------------------
    data_cfg = cfg.get("data", {})
    class_names = data_cfg.get("class_names", ["positive", "negative"])
    positive_name = data_cfg.get("positive_class", class_names[0])

    labels_map: Dict[str, Any] = {}

    # IMPORTANT: Both label loaders are called only after inference CSVs are written above.
    if args.labels_csv is not None:
        labels_map.update(load_labels_csv(args.labels_csv))
    if args.labels_from_folders:
        labels_map.update(infer_labels_from_folders(fov_paths=fov_paths, class_names=class_names))

    if not labels_map:
        print("[INFO] No labels provided. Inference complete; skipping metrics.")
        return

    # Read predictions from disk to preserve post-hoc separation
    fov_df = pd.read_csv(fovs_csv)
    fov_df["true_label_raw"] = fov_df["fov_id"].map(labels_map)
    fov_df["y_true"] = fov_df["true_label_raw"].apply(lambda v: normalize_true_label(v, positive_class_name=positive_name))
    eval_df = fov_df.dropna(subset=["y_true"]).copy()
    if len(eval_df) == 0:
        print("[WARN] Labels provided but none matched fov_id entries. Skipping metrics.")
        return

    y_true = eval_df["y_true"].astype(int).to_numpy()
    score_cols = [c for c in eval_df.columns if c.startswith("FOV_score_")]

    print(f"[INFO] Post-hoc eval on {len(eval_df)} labeled FOVs. Score columns: {score_cols}")

    # Quick single-threshold summary at threshold_T
    T0 = float(args.threshold_T)
    n_neg = max(1, int((y_true == 0).sum()))
    for col in score_cols:
        scores = eval_df[col].astype(float).to_numpy()
        y_pred = (scores >= T0).astype(int)
        tp, fp, tn, fn = _confusion_from_preds(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        fp_per_neg = fp / n_neg
        print(
            f"[EVAL @T={T0:g}] {col}: sens={sens:.4f} spec={spec:.4f} FP/neg={fp_per_neg:.4f} "
            f"(TP={tp} FP={fp} TN={tn} FN={fn})"
        )

    # Multi-hit quick summary at threshold_T (recomputed from per-tile probs for correctness)
    tile_df = pd.read_csv(tiles_csv)
    tile_df = tile_df[tile_df["fov_id"].isin(eval_df["fov_id"].tolist())].copy()
    grouped = tile_df.groupby("fov_id")["predicted_probability"].apply(lambda s: s.astype(float).to_numpy()).to_dict()

    # Build tile_probs_list aligned with eval_df row order
    fov_ids_ordered = eval_df["fov_id"].tolist()
    tile_probs_list: List[np.ndarray] = []
    missing_tiles = 0
    for fid in fov_ids_ordered:
        arr = grouped.get(fid)
        if arr is None:
            missing_tiles += 1
            arr = np.asarray([], dtype=np.float32)
        tile_probs_list.append(arr.astype(np.float32))

    if missing_tiles > 0:
        print(f"[WARN] {missing_tiles} labeled FOVs had no tile rows in per_tile.csv. Multi-hit eval may be off.")

    for N in args.multi_hit_N:
        Nn = int(N)
        preds = []
        for probs_arr in tile_probs_list:
            preds.append(int(np.sum(probs_arr >= T0) >= Nn))
        y_pred = np.asarray(preds, dtype=np.int32)
        tp, fp, tn, fn = _confusion_from_preds(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        fp_per_neg = fp / n_neg
        print(
            f"[EVAL @T={T0:g}] MULTI_HIT_N{Nn}: sens={sens:.4f} spec={spec:.4f} FP/neg={fp_per_neg:.4f} "
            f"(TP={tp} FP={fp} TN={tn} FN={fn})"
        )

    if args.sweep:
        thresholds = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
        all_results: List[SweepResult] = []

        # Sweep score-based methods
        for col in score_cols:
            scores = eval_df[col].astype(float).to_numpy().astype(np.float32)
            all_results.extend(
                sweep_thresholds_for_scores(y_true=y_true, scores=scores, method_name=col, thresholds=thresholds)
            )

        # Sweep multi-hit methods
        for N in args.multi_hit_N:
            Nn = int(N)
            method_name = f"MULTI_HIT_N{Nn}"
            all_results.extend(
                sweep_thresholds_for_multihit(
                    y_true=y_true, tile_probs_list=tile_probs_list, N=Nn, method_name=method_name, thresholds=thresholds
                )
            )

        sweep_path = out_dir / "threshold_sweep_metrics.csv"
        sweep_df = pd.DataFrame([r.__dict__ for r in all_results])
        sweep_df.to_csv(sweep_path, index=False)
        print(f"[OK] Wrote threshold sweep metrics: {sweep_path}")

        if sweep_df.empty or "method" not in sweep_df.columns:
            print("[INFO] Threshold sweep produced no rows (need both positive and negative labeled FOVs). Done.")
            return

        # Pick best threshold at/above target sensitivity: minimize FP/neg (tie-breaker: lower threshold)
        target = float(args.target_sensitivity)
        methods = sorted(sweep_df["method"].unique().tolist())
        for m in methods:
            sub = sweep_df[sweep_df["method"] == m].copy()
            sub = sub[sub["sensitivity"] >= target]
            if len(sub) == 0:
                print(f"[WARN] {m}: no threshold achieved sensitivity >= {target:.3f}")
                continue
            best = sub.sort_values(["fp_per_negative", "threshold"], ascending=[True, True]).iloc[0]
            print(
                f"[BEST @sens>={target:.3f}] {m}: "
                f"T={best['threshold']:.3f} sens={best['sensitivity']:.4f} "
                f"spec={best['specificity']:.4f} FP/neg={best['fp_per_negative']:.4f}"
            )


if __name__ == "__main__":
    main()
