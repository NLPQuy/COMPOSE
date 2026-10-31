"""Part-level matching on projected slots (Phase II, Sec. 4.2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def project_slots(proj: torch.nn.Module, slots: torch.Tensor, center: bool = True) -> torch.Tensor:
    """``z_k = L2norm(W2 phi_k)`` (Eq. 3), then per-image centering ``z_hat_k`` (Eq. 6)."""
    n, k, d = slots.shape
    z = F.normalize(proj(slots.reshape(n * k, d)).reshape(n, k, -1), dim=-1)
    if not center:
        return z
    return F.normalize(z - z.mean(dim=1, keepdim=True), dim=-1)


def top_slots(z: torch.Tensor, omega: torch.Tensor, kappa: int) -> torch.Tensor:
    """The ``kappa`` slots of each image with the largest router weight ``omega``."""
    idx = omega.topk(min(kappa, z.shape[1]), dim=1).indices
    return torch.gather(z, 1, idx.unsqueeze(-1).expand(-1, -1, z.shape[-1]))


def _class_slots(sup_top: torch.Tensor, sup_lbl: torch.Tensor, c: int) -> torch.Tensor:
    """Union over the support images of class ``c`` of their top slots, ``[P, D]``."""
    return sup_top[sup_lbl == c].reshape(-1, sup_top.shape[-1])


def chamfer(qry_top: torch.Tensor, cls_slots: torch.Tensor) -> torch.Tensor:
    """Bidirectional Chamfer similarity of every query to one class, ``[Q]``.

    Mean of the query-to-class and class-to-query terms (Eq. 8).
    """
    sim = torch.einsum("qkd,pd->qkp", qry_top, cls_slots)
    return 0.5 * (sim.max(dim=-1).values.mean(dim=-1) + sim.max(dim=-2).values.mean(dim=-1))


def hungarian(qry_top: torch.Tensor, cls_slots: torch.Tensor) -> torch.Tensor:
    """One-to-one assignment instead of Chamfer's max (ablation "COMPOSE-Hungarian")."""
    from scipy.optimize import linear_sum_assignment

    sim = torch.einsum("qkd,pd->qkp", qry_top, cls_slots).cpu().numpy()
    out = torch.zeros(sim.shape[0])
    for q in range(sim.shape[0]):
        r, c = linear_sum_assignment(sim[q], maximize=True)
        out[q] = float(sim[q][r, c].mean())
    return out


MATCHERS = {"chamfer": chamfer, "hungarian": hungarian}


def part_scores(
    qry_top: torch.Tensor,
    sup_top: torch.Tensor,
    sup_lbl: torch.Tensor,
    n_way: int,
    matcher: str = "chamfer",
) -> torch.Tensor:
    """``[Q, n_way]`` part-level similarity of each query to each class."""
    fn = MATCHERS[matcher]
    scores = torch.zeros(qry_top.shape[0], n_way, device=qry_top.device)
    for c in range(n_way):
        scores[:, c] = fn(qry_top, _class_slots(sup_top, sup_lbl, c))
    return scores
