"""MLP slot router and shared projection head (Phase I, Sec. 4.1).

``phi`` is always ``phi_full [N, K+1, D]``: row 0 is the CLS token, which the router
never reads; rows 1..K are the slots.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def slots_of(phi: torch.Tensor) -> torch.Tensor:
    return phi[:, 1:]


def prototype_logits(emb: torch.Tensor, sup_lbl: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """``tau * cos(e_q, P_c)``; ``emb`` holds the support rows first, then the queries."""
    n_sup = sup_lbl.shape[0]
    n_way = int(sup_lbl.max().item()) + 1
    protos = F.normalize(
        torch.stack([emb[:n_sup][sup_lbl == c].mean(dim=0) for c in range(n_way)]),
        dim=-1,
    )
    return (emb[n_sup:] @ protos.T) * tau


class SlotRouter(nn.Module):
    """COMPOSE router: ``omega = softmax_k(v^T relu(W1 phi_k))`` (Eq. 2) and ``W2`` (Eqs. 3-4).

    Trained with holistic prototype cross-entropy plus the cross-correlation penalty
    ``lambda_d * ||C - I||_F^2 / D``; no part-level matching enters the loss.
    """

    def __init__(self, phi_dim: int, hidden_dim: int = 64, lambda_d: float = 0.02):
        super().__init__()
        self.lambda_d = lambda_d
        self.hidden = nn.Linear(phi_dim, hidden_dim)       # W1
        self.score = nn.Linear(hidden_dim, 1)              # v
        self.proj = nn.Linear(phi_dim, phi_dim)            # W2, initialised as the identity
        with torch.no_grad():
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.log_tau = nn.Parameter(torch.tensor(2.3))

    # -- routing -----------------------------------------------------------------
    def _slot_features(self, slots: torch.Tensor) -> torch.Tensor:
        return F.relu(self.hidden(slots))

    def route(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(slots [N, K, D], omega [N, K])``."""
        slots = slots_of(phi)
        omega = F.softmax(self.score(self._slot_features(slots)).squeeze(-1), dim=-1)
        return slots, omega

    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1.0)

    def embed(self, phi: torch.Tensor) -> torch.Tensor:
        """Holistic embedding ``e = L2norm(W2 sum_k omega_k phi_k)`` (Eq. 4)."""
        slots, omega = self.route(phi)
        return F.normalize(self.proj((omega.unsqueeze(-1) * slots).sum(dim=1)), dim=-1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.embed(phi)

    # -- training ----------------------------------------------------------------
    def cross_correlation_penalty(self, slots: torch.Tensor) -> torch.Tensor:
        """``||C - I||_F^2 / D`` over the batch of unnormalised slot projections ``W2 phi_k``."""
        n, k, d = slots.shape
        y = self.proj(slots.reshape(n * k, d))
        y = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-5)
        corr = (y.T @ y) / y.shape[0]
        eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
        return ((corr - eye) ** 2).sum() / corr.shape[0]

    def episode_logits(self, sup_phi, sup_lbl, qry_phi) -> tuple[torch.Tensor, torch.Tensor]:
        """Training logits for the queries and the weighted regulariser."""
        slots, omega = self.route(torch.cat([sup_phi, qry_phi], dim=0))
        emb = F.normalize(self.proj((omega.unsqueeze(-1) * slots).sum(dim=1)), dim=-1)
        logits = prototype_logits(emb, sup_lbl, self.tau())
        return logits, self.lambda_d * self.cross_correlation_penalty(slots)


class BiGRURouter(SlotRouter):
    """Ablation "GRU-router": slot importance from a BiGRU over the K slots instead of W1.

    ``hidden`` and the first ``score`` layer are still created by the parent so the
    parameter initialisation order matches the runs reported in the paper; ``hidden``
    is unused.
    """

    def __init__(self, phi_dim: int, hidden_dim: int = 64, lambda_d: float = 0.02):
        super().__init__(phi_dim, hidden_dim=hidden_dim, lambda_d=lambda_d)
        self.gru = nn.GRU(phi_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.score = nn.Linear(2 * hidden_dim, 1)

    def _slot_features(self, slots: torch.Tensor) -> torch.Tensor:
        return self.gru(slots)[0]
