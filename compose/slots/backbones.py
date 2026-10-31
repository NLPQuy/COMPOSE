"""Other patch sources for the slot readout (Tables 2, 7, 8, 23 and 25).

The DINOSAUR attention maps stay those of the released model; only the tokens they
average change. For a backbone with the same patch grid as the DINOSAUR backbone::

    phi_k = sum_p alpha_kp / (sum_p' alpha_kp' + 1e-8) * token_new[p],   cls = CLS token of the new backbone

DINOSAUR with DINOv2 ViT-B/14 (256 patches at 224) serves the /14 backbones; a DINOSAUR
trained with DINOv1 ViT-B/16 (196 patches) serves the /16 ones. SAM runs at 1024 and its
64x64 token grid is average-pooled to 14x14 (or the attention is upsampled to 64x64).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PatchSource:
    """One backbone used as patch source (``configs/backbones/*.yaml``).

    ``model``: timm name. ``checkpoint``: local weights (None = timm pretrained weights).
    ``layer``: block index to read (None = ``forward_features``, i.e. after the final norm).
    ``layer_norm``: apply the final norm to an intermediate block. ``renorm``: feed the
    backbone its own mean/std instead of ImageNet's. ``input_size``: resolution of the
    backbone (attention is always computed at 224). ``pool_to``: pool the token grid to this
    size (SAM). ``dinosaur``: ``b14`` or ``b16``. ``kind``: ``vit`` or ``sam``.
    """

    name: str
    model: str
    dinosaur: str = "b14"
    checkpoint: str | None = None
    layer: int | None = None
    layer_norm: bool = False
    renorm: bool = False
    input_size: int = 224
    pool_to: tuple[int, int] | None = None
    kind: str = "vit"

    @classmethod
    def from_yaml(cls, path: str | Path, checkpoint: str | None = None) -> "PatchSource":
        raw = yaml.safe_load(Path(path).read_text())
        raw.pop("train", None)                       # router overrides, read by scripts/run.py
        if raw.get("pool_to") is not None:
            raw["pool_to"] = tuple(raw["pool_to"])
        if checkpoint is not None:
            raw["checkpoint"] = checkpoint
        return cls(**raw)


# -- model construction ---------------------------------------------------------------
def _register_mae_vith14() -> None:
    """MAE ViT-H/14 is not in timm; same definition as ``videosaur/modules/timm.py``."""
    import timm
    from timm.models import vision_transformer

    if "vit_huge_patch14_224_mae" in timm.list_models("vit_huge_patch14_224_mae"):
        return

    @timm.models.register_model
    def vit_huge_patch14_224_mae(pretrained: bool = False, **kwargs):
        args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)
        args.update(kwargs)
        return vision_transformer._create_vision_transformer("vit_huge_patch14_224_mae", pretrained=pretrained, **args)


def _checkpoint_state(path: str, kind: str) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "sam":
        state = {k[len("image_encoder."):]: v for k, v in raw.items() if k.startswith("image_encoder.")} or raw
        return {k.replace("mlp.lin1", "mlp.fc1").replace("mlp.lin2", "mlp.fc2"): v for k, v in state.items()}
    for key in ("teacher", "student", "state_dict", "model", "encoder"):       # iBOT / DINO / MAE layouts
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
            break
    return {k.removeprefix("backbone.").removeprefix("module."): v for k, v in raw.items()}


def build_backbone(src: PatchSource) -> nn.Module:
    import timm

    if "mae" in src.model:
        _register_mae_vith14()
    model = timm.create_model(src.model, pretrained=src.checkpoint is None, num_classes=0)
    if src.checkpoint is not None:
        missing, unexpected = model.load_state_dict(_checkpoint_state(src.checkpoint, src.kind), strict=False)
        if src.kind == "sam" and (missing or unexpected):
            raise RuntimeError(f"SAM weights did not load cleanly: missing {missing[:5]}, unexpected {unexpected[:5]}")
        broken = [k for k in missing if k.startswith("blocks.")]
        if broken:
            raise RuntimeError(f"{src.model}: checkpoint misses transformer weights, e.g. {broken[:3]}")
    return model.eval().requires_grad_(False)


# -- tokens ---------------------------------------------------------------------------
class _Stop(Exception):
    pass


def _renorm(images: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
    cfg = getattr(backbone, "pretrained_cfg", None) or {}
    mean, std = tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD))
    if mean == IMAGENET_MEAN and std == IMAGENET_STD:
        return images
    v = lambda x: torch.tensor(x, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    return ((images * v(IMAGENET_STD) + v(IMAGENET_MEAN)) - v(mean)) / v(std)


@torch.no_grad()
def backbone_tokens(src: PatchSource, backbone: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``(cls [B, D] or None, patches [B, P, D])`` of the new backbone."""
    if src.kind == "sam":
        _, inter = backbone.forward_intermediates(images, indices=[src.layer if src.layer is not None else 11])
        feat = inter[0]                                            # [B, D, H, W], before the neck
        if src.pool_to is not None:
            feat = F.adaptive_avg_pool2d(feat, src.pool_to)
        return None, feat.flatten(2).transpose(1, 2)

    if src.renorm:
        images = _renorm(images, backbone)
    if src.layer is None:
        tokens = backbone.forward_features(images)
    else:
        captured = []

        def hook(_m, _i, out):
            captured.append(out[0] if isinstance(out, tuple) else out)
            raise _Stop

        handle = backbone.blocks[src.layer].register_forward_hook(hook)
        try:
            backbone.forward_features(images)
        except _Stop:
            pass
        finally:
            handle.remove()
        tokens = backbone.norm(captured[0]) if src.layer_norm else captured[0]
    n_prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    return tokens[:, 0], tokens[:, n_prefix:]


def upsample_attention(attn: torch.Tensor, target_p: int) -> torch.Tensor:
    """Bilinear resize of ``[B, K, P]`` attention to ``target_p`` patches (square grids)."""
    if attn.shape[-1] == target_p:
        return attn
    b, k, p = attn.shape
    src, dst = int(round(math.sqrt(p))), int(round(math.sqrt(target_p)))
    up = F.interpolate(attn.reshape(b, k, src, src), size=(dst, dst), mode="bilinear", align_corners=False)
    return up.reshape(b, k, target_p)


@torch.no_grad()
def readout(src: PatchSource, backbone: nn.Module, dino_out: dict, images: torch.Tensor,
            images_src: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Replace ``phi`` / ``phi_tilde`` / ``cls`` of a DINOSAUR output with the new backbone's tokens."""
    cls, patches = backbone_tokens(src, backbone, images if images_src is None else images_src)
    attn = dino_out["attn_maps"]
    if src.kind == "sam" and src.pool_to is None:
        attn = upsample_attention(attn, patches.shape[1])
    if patches.shape[1] != attn.shape[-1]:
        raise RuntimeError(f"patch grids differ: backbone {patches.shape[1]}, DINOSAUR attention {attn.shape[-1]}")
    weights = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
    phi = torch.einsum("bkp,bpd->bkd", weights, patches)
    out = {**dino_out, "phi": phi, "phi_tilde": F.normalize(phi, dim=-1)}
    if cls is None:
        out.pop("cls")                                             # SAM has no CLS token
    else:
        out["cls"] = cls
    return out
