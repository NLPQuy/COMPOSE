"""Slot features on an adapted backbone.

DINOSAUR supplies the slot attention maps; the patch and CLS tokens come from the
base-session-adapted ViT (after its final norm). For each image::

    phi_full[0]  = L2norm(CLS)
    phi_full[k]  = L2norm(sum_p attn[k, p] * token[p])     k = 1..K
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from compose.cil.adapter import AdapterViT
from compose.slots.dinosaur import DINOSAUR


@torch.no_grad()
def extract(dataset, vit: AdapterViT, dinosaur: DINOSAUR, *, device, generator: torch.Generator,
            batch_size: int = 64, num_workers: int = 4, desc: str = "") -> tuple[torch.Tensor, torch.Tensor]:
    """``(phi_full [N, K+1, D], labels [N])`` for every image of ``dataset``, in order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    phis, labels = [], []
    for _, images, y in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        tokens = vit.tokens(images)
        attn = dinosaur(images, generator=generator)["attn_maps"]        # [B, K, P]
        slots = F.normalize(torch.bmm(attn, tokens[:, 1:]), dim=-1)
        phis.append(torch.cat([F.normalize(tokens[:, 0], dim=-1).unsqueeze(1), slots], dim=1).cpu())
        labels.append(torch.as_tensor(y).long())
    return torch.cat(phis), torch.cat(labels)
