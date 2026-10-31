"""Base-session adaptation of the backbone (Table 11), as in LAMDA-PILOT's RanPAC learner.

Before the class-incremental sessions start, the DINOv2 backbone is adapted to the base
classes: AdaptFormer adapters plus a cosine classifier are trained with cross-entropy on
the base session, then the adapted backbone is frozen for every later session. All
methods in Table 11 share this step, so none gets a better-adapted backbone than another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compose.cil.adapter import AdapterViT, dinov2_adapter_vit


@dataclass(frozen=True)
class BaseSessionConfig:
    """LAMDA-PILOT ``exps/ranpac.json`` with the DINOv2 adapter backbone."""

    epochs: int = 20
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    min_lr: float = 0.0
    batch_size: int = 48
    ffn_num: int = 64
    num_workers: int = 8


class CosineLinear(nn.Module):
    """``sigma * cos(x, w_c)`` classifier (LAMDA-PILOT ``backbone/linears.py``)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma = nn.Parameter(torch.empty(1))
        self.weight.data.uniform_(-1.0 / math.sqrt(in_features), 1.0 / math.sqrt(in_features))
        self.sigma.data.fill_(1)

    def forward(self, x):
        return self.sigma * F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))


class AdaptedNet(nn.Module):
    def __init__(self, backbone: AdapterViT, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = CosineLinear(backbone.out_dim, n_classes)

    def forward(self, x):
        return self.fc(self.backbone(x))


def adapt_backbone(train_set, n_classes: int, cfg: BaseSessionConfig, device: torch.device,
                   backbone: AdapterViT | None = None) -> AdapterViT:
    """Train adapters + classifier on the base session; return the frozen adapted backbone."""
    net = AdaptedNet(backbone or dinov2_adapter_vit(cfg.ffn_num), n_classes).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.min_lr)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    for epoch in range(cfg.epochs):
        net.train()
        total_loss, correct, seen = 0.0, 0, 0
        for _, images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += int((logits.argmax(1) == targets).sum())
            seen += len(targets)
        sched.step()
        print(f"  base session epoch {epoch + 1}/{cfg.epochs}  loss={total_loss / len(loader):.3f}  "
              f"train acc={100.0 * correct / seen:.2f}", flush=True)

    backbone = net.backbone.eval()
    backbone.requires_grad_(False)
    return backbone
