"""Frozen DINOSAUR (DINOv2 ViT-B/14 + slot attention) and the slot readout of Eq. 1.

A dependency-light re-implementation of the inference path of the official DINOSAUR
model in ``videosaur`` (Seitzer et al., Zadaianchuk et al.; MIT licence), loading the
released COCO checkpoint ``coco_dinosaur_base14_dinov2.ckpt`` unchanged. Only ``timm``
is needed; ``tests/`` checks it against the original modules.

For each image it returns the backbone patch tokens ``F_n`` (output of block 12, before
the final norm), the CLS token of the same layer, the slot-attention masks
``alpha_kn`` (softmax over slots) and

    phi_k = sum_n alpha_kn / (sum_n' alpha_kn' + 1e-8) * F_n,     phi_tilde_k = phi_k / ||phi_k||.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_URL = (
    "https://huggingface.co/andriizadaianchuk/dinosaur_dinov2/resolve/main/coco_dinosaur_base14_dinov2.ckpt"
)
CHECKPOINT_SHA256 = "a3353f85aef338762bc75606bfbaa4c9817b128c60f72ca740935c338382f213"


class _MLP(nn.Module):
    """``[LayerNorm] -> Linear -> ReLU -> Linear [+ residual]`` (``videosaur.modules.networks.MLP``)."""

    def __init__(self, inp: int, hidden: int, out: int, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.layers = nn.Sequential(nn.LayerNorm(inp), nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x):
        y = self.layers(x)
        return x + y if self.residual else y


class SlotAttention(nn.Module):
    """Slot attention with GRU update and residual MLP (Locatello et al., 2020)."""

    def __init__(self, dim: int = 256, n_iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = _MLP(dim, 4 * dim, dim, residual=True)
        self.norm_features = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.n_iters = n_iters
        self.eps = eps
        self.scale = dim ** -0.5

    def forward(self, slots: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(slots [B, K, D], masks [B, K, N])``; masks are the last softmax over slots."""
        features = self.norm_features(features)
        keys, values = self.to_k(features), self.to_v(features)
        for _ in range(self.n_iters):
            normed = self.norm_slots(slots)
            dots = torch.einsum("bsd,bfd->bsf", self.to_q(normed), keys) * self.scale
            masks = torch.softmax(dots, dim=1)
            attn = masks + self.eps
            attn = attn / attn.sum(-1, keepdim=True)
            updates = torch.einsum("bsf,bfd->bsd", attn, values)
            slots = self.gru(updates.flatten(0, 1), normed.flatten(0, 1)).unflatten(0, normed.shape[:2])
            slots = self.mlp(slots)
        return slots, masks


class DINOSAUR(nn.Module):
    def __init__(self, n_slots: int = 7, slot_dim: int = 256, backbone: str = "vit_base_patch14_dinov2"):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=False, dynamic_img_size=True, num_classes=0)
        feat_dim = self.backbone.embed_dim
        self.output_transform = _MLP(feat_dim, 2 * feat_dim, slot_dim)
        self.slot_mean = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slot_log_std = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slot_attention = SlotAttention(slot_dim)
        self.n_slots = n_slots
        self.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path: str | Path, n_slots: int = 7,
                        backbone: str = "vit_base_patch14_dinov2") -> "DINOSAUR":
        """``backbone``: timm name of the ViT inside the checkpoint (``vit_base_patch16_224.dino``
        for a DINOSAUR trained on DINOv1 ViT-B/16)."""
        sd = torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"]
        model = cls(n_slots=n_slots, backbone=backbone)
        prefixes = {
            "encoder.backbone.model.": model.backbone,
            "encoder.output_transform.": model.output_transform,
            "processor.corrector.": model.slot_attention,
        }
        for prefix, module in prefixes.items():
            part = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            missing, unexpected = module.load_state_dict(part, strict=False)
            # The checkpoint keeps the ViT only up to block 12: no final norm, no head.
            missing = [k for k in missing if not k.startswith(("norm.", "head", "fc_norm"))]
            if missing or unexpected:
                raise RuntimeError(f"{prefix}: missing {missing}, unexpected {unexpected}")
        with torch.no_grad():
            model.slot_mean.copy_(sd["initializer.mean"])
            model.slot_log_std.copy_(sd["initializer.log_std"])
        return model.eval()

    def backbone_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cls [B, D], patches [B, N, D])`` after the last transformer block, before the final norm."""
        vit = self.backbone
        x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(images))))
        for block in vit.blocks:
            x = block(x)
        return x[:, 0], x[:, vit.num_prefix_tokens:]

    @torch.no_grad()
    def forward(self, images: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        cls, tokens = self.backbone_tokens(images)
        features = self.output_transform(tokens)
        noise = torch.randn(
            images.shape[0], self.n_slots, self.slot_mean.shape[-1], generator=generator, device="cpu"
        ).to(images.device)
        slots, masks = self.slot_attention(self.slot_mean + noise * self.slot_log_std.exp(), features)

        weights = masks / (masks.sum(dim=-1, keepdim=True) + 1e-8)
        phi = torch.einsum("bkp,bpd->bkd", weights, tokens)
        return {
            "phi": phi,
            "phi_tilde": F.normalize(phi, dim=-1),
            "cls": cls,
            "slots": slots,
            "attn_maps": masks,
            "attn_mass": masks.sum(dim=-1),
            "attn_entropy": -(masks * (masks + 1e-10).log()).sum(dim=-1),
        }
