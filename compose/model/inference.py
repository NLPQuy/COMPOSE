"""Phase II: gradient-free compositional inference (Sec. 4.2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from compose.model.matching import part_scores, project_slots, top_slots
from compose.model.router import SlotRouter, prototype_logits


@dataclass(frozen=True)
class InferenceConfig:
    """How Phase II reads one trained router.

    ``gamma``: holistic weight in Eq. 9 (Chamfer gets ``1 - gamma``).
    ``kappa``: query/support slots kept by router weight.
    ``center``: per-image slot centering (Eq. 6); off for the "w/o centering" ablation.
    ``pool``: ``router`` (Eq. 4) or ``mean`` (holistic branch averages the slots).
    ``matcher``: ``chamfer`` (Eq. 8) or ``hungarian``.
    """

    name: str = "COMPOSE"
    gamma: float = 0.3
    kappa: int = 4
    center: bool = True
    pool: str = "router"
    matcher: str = "chamfer"


def to_local(labels: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    classes = sorted(int(c) for c in labels.unique().tolist())
    index = {c: i for i, c in enumerate(classes)}
    return torch.tensor([index[int(c)] for c in labels.tolist()], dtype=torch.long, device=labels.device), classes


@torch.no_grad()
def compose_scores(
    model: SlotRouter,
    sup_phi: torch.Tensor,
    sup_lbl: torch.Tensor,
    qry_phi: torch.Tensor,
    cfg: InferenceConfig = InferenceConfig(),
) -> tuple[torch.Tensor, list[int]]:
    """``s(q, c) = gamma * s_hol + (1 - gamma) * tau * s_part`` (Eq. 9).

    Returns ``[Q, n_way]`` scores and the class id of each column.
    """
    model.eval()
    device = next(model.parameters()).device
    sup_phi, qry_phi, sup_lbl = sup_phi.to(device), qry_phi.to(device), sup_lbl.to(device)
    sup_local, classes = to_local(sup_lbl)
    n_sup = sup_phi.shape[0]
    all_phi = torch.cat([sup_phi, qry_phi], dim=0)

    slots, omega = model.route(all_phi)
    if cfg.pool == "router":
        emb = model.embed(all_phi)
    elif cfg.pool == "mean":
        emb = F.normalize(model.proj(slots.mean(dim=1)), dim=-1)
    else:
        raise ValueError(f"unknown pool {cfg.pool!r}")
    tau = model.tau()
    holistic = prototype_logits(emb, sup_local, tau)

    z = project_slots(model.proj, slots, center=cfg.center)
    sup_top = top_slots(z[:n_sup], omega[:n_sup], cfg.kappa)
    qry_top = top_slots(z[n_sup:], omega[n_sup:], cfg.kappa)
    part = part_scores(qry_top, sup_top, sup_local, len(classes), cfg.matcher) * tau
    return cfg.gamma * holistic + (1.0 - cfg.gamma) * part, classes


def predict(model: SlotRouter, sup_phi, sup_lbl, qry_phi, cfg: InferenceConfig = InferenceConfig()) -> torch.Tensor:
    """Predicted class id of each query."""
    scores, classes = compose_scores(model, sup_phi, sup_lbl, qry_phi, cfg)
    return torch.tensor([classes[i] for i in scores.argmax(dim=1).cpu().tolist()], dtype=torch.long)
