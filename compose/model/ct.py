"""COMPOSE-CT: the ablation that puts Chamfer matching into the training loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from compose.model.matching import chamfer, project_slots, top_slots
from compose.model.router import SlotRouter, prototype_logits


class ChamferTrainedRouter(SlotRouter):
    """Router trained on ``w * holistic + (1 - w) * Chamfer`` logits.

    ``holistic_weight`` 1.0 recovers COMPOSE, 0.3 is COMPOSE-CT and 0.0 trains on
    Chamfer alone. During training the query keeps its top-``kappa`` slots and each
    class uses all slots of its support images. Inference is the same as COMPOSE.
    """

    def __init__(
        self,
        phi_dim: int,
        hidden_dim: int = 64,
        lambda_d: float = 0.02,
        holistic_weight: float = 0.3,
        kappa: int = 4,
    ):
        super().__init__(phi_dim, hidden_dim=hidden_dim, lambda_d=lambda_d)
        self.holistic_weight = holistic_weight
        self.kappa = kappa

    def episode_logits(self, sup_phi, sup_lbl, qry_phi) -> tuple[torch.Tensor, torch.Tensor]:
        n_sup = sup_phi.shape[0]
        slots, omega = self.route(torch.cat([sup_phi, qry_phi], dim=0))
        emb = F.normalize(self.proj((omega.unsqueeze(-1) * slots).sum(dim=1)), dim=-1)
        tau = self.tau()
        holistic = prototype_logits(emb, sup_lbl, tau)

        z = project_slots(self.proj, slots)
        qry_top = top_slots(z[n_sup:], omega[n_sup:], self.kappa)
        sup_z = z[:n_sup]
        n_way = int(sup_lbl.max().item()) + 1
        part = torch.zeros(qry_top.shape[0], n_way, device=z.device)
        for c in range(n_way):
            part[:, c] = chamfer(qry_top, sup_z[sup_lbl == c].reshape(-1, sup_z.shape[-1]))
        part = part * tau

        logits = self.holistic_weight * holistic + (1.0 - self.holistic_weight) * part
        return logits, self.lambda_d * self.cross_correlation_penalty(slots)
