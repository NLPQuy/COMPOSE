"""DINOv2 ViT-B/14 with AdaptFormer adapters, for the base-session adaptation of Table 11.

Ported from ``backbone/vit_adapter.py`` of LAMDA-PILOT (Sun et al., MIT licence), which
follows https://github.com/jxhe/unify-parameter-efficient-tuning. Parameter names are kept,
so state dicts are interchangeable with LAMDA-PILOT. Only the adapters are trainable.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Parallel bottleneck adapter, LoRA-style init (up projection starts at zero)."""

    def __init__(self, dim: int = 768, bottleneck: int = 64, scale: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.down_proj = nn.Linear(dim, bottleneck)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, dim)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = nn.functional.dropout(self.non_linear_func(self.down_proj(x)), p=self.dropout, training=self.training)
        return self.up_proj(down) * self.scale


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def _heads(self, t: torch.Tensor, b: int) -> torch.Tensor:
        return t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(
            b * self.num_heads, -1, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self._heads(self.q_proj(x), b), self._heads(self.k_proj(x), b), self._heads(self.v_proj(x), b)
        attn = nn.functional.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.bmm(self.attn_drop(attn), v).view(b, self.num_heads, n, self.head_dim).transpose(1, 2)
        return self.proj_drop(self.proj(out.reshape(b, n, c)))


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(1e-5 * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_num: int):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = norm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()
        self.mlp_drop = nn.Dropout(0.0)
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)
        self.adaptmlp = Adapter(dim, ffn_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        adapt = self.adaptmlp(x)
        residual = x
        x = self.drop_path(self.mlp_drop(self.fc2(self.mlp_drop(self.act(self.fc1(self.norm2(x)))))))
        return residual + self.ls2(x + adapt)


class AdapterViT(nn.Module):
    """ViT-B/14 at 224x224 (256 patches). ``forward`` returns the normalised CLS token."""

    def __init__(self, ffn_num: int = 64, img_size: int = 224, patch_size: int = 14,
                 dim: int = 768, depth: int = 12, num_heads: int = 12):
        super().__init__()
        from timm.layers import PatchEmbed

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.Sequential(*[Block(dim, num_heads, ffn_num) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.out_dim = dim

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """All tokens after the final norm, ``[B, 1 + N, D]`` (CLS first)."""
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return self.norm(self.blocks(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokens(x)[:, 0]


def dinov2_adapter_vit(ffn_num: int = 64) -> AdapterViT:
    """Load DINOv2 ViT-B/14 (LVD-142M) from timm into :class:`AdapterViT`; freeze all but adapters."""
    import timm

    model = AdapterViT(ffn_num=ffn_num)
    sd = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).state_dict()
    for key in list(sd):
        if key.endswith(("qkv.weight", "qkv.bias")):
            q, k, v = sd.pop(key).chunk(3)
            for name, t in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                sd[key.replace("qkv", name)] = t
        elif "mlp.fc" in key:
            sd[key.replace("mlp.", "")] = sd.pop(key)
    missing, _ = model.load_state_dict(sd, strict=False)
    for name, p in model.named_parameters():
        p.requires_grad = name in missing
    if any("adaptmlp" not in k for k in missing):
        raise RuntimeError(f"DINOv2 weights did not cover the backbone: {[k for k in missing if 'adaptmlp' not in k]}")
    return model
