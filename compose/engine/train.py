"""Episodic meta-training of the router on one continual session."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from compose.data.sampling import Rows
from compose.data.slot_store import SlotStore
from compose.model.router import SlotRouter


def sample_training_episode(
    labels: torch.Tensor, *, n_way: int, n_support: int, n_query: int, rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(support idx, support labels, query idx, query labels)``, labels relative to the episode.

    Indices are positions in ``labels``. A class with fewer than ``n_support + n_query``
    images is sampled with replacement.
    """
    classes = labels.unique()
    n_way = min(n_way, len(classes))
    chosen = classes[torch.randperm(len(classes), generator=rng)[:n_way]]
    total = n_support + n_query
    sup_idx, qry_idx = [], []
    for c in chosen.tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if len(idx) >= total:
            picked = idx[torch.randperm(len(idx), generator=rng)[:total]]
        else:
            picked = idx[torch.randint(len(idx), (total,), generator=rng)]
        sup_idx.append(picked[:n_support])
        qry_idx.append(picked[n_support:])
    rel = torch.arange(n_way, dtype=torch.long)
    return (
        torch.cat(sup_idx), rel.repeat_interleave(n_support),
        torch.cat(qry_idx), rel.repeat_interleave(n_query),
    )


def meta_train(
    model: SlotRouter,
    store: SlotStore,
    rows: Rows,
    *,
    n_way: int,
    n_support: int,
    n_query: int,
    n_episodes: int,
    lr: float,
    seed: int,
    device: torch.device,
    log_every: int | None = None,
) -> SlotRouter:
    """Adam + cosine schedule over ``n_episodes`` episodes drawn from ``rows``."""
    if n_episodes <= 0:
        return model.eval()
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes)
    labels = torch.as_tensor(rows.labels, dtype=torch.long)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    log_every = log_every or max(1, n_episodes // 5)

    for ep in range(n_episodes):
        sup_idx, sup_lbl, qry_idx, qry_lbl = sample_training_episode(
            labels, n_way=n_way, n_support=n_support, n_query=n_query, rng=rng
        )
        sup_phi = store.gather(rows.take(sup_idx.numpy())).to(device)
        qry_phi = store.gather(rows.take(qry_idx.numpy())).to(device)
        sup_lbl, qry_lbl = sup_lbl.to(device), qry_lbl.to(device)

        logits, reg = model.episode_logits(sup_phi, sup_lbl, qry_phi)
        loss = F.cross_entropy(logits, qry_lbl) + reg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if (ep + 1) % log_every == 0:
            acc = (logits.argmax(dim=1) == qry_lbl).float().mean().item() * 100.0
            print(f"    episode {ep + 1}/{n_episodes}  loss={loss.item():.4f}  acc={acc:.2f}", flush=True)
    return model.eval()


def exemplar_rows(rows: Rows, shots: int, seed: int) -> Rows:
    """Up to ``shots`` random images per class of a session (the replay buffer)."""
    labels = torch.as_tensor(rows.labels, dtype=torch.long)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    picked = []
    for c in sorted(int(x) for x in labels.unique().tolist()):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        picked.append(idx[torch.randperm(len(idx), generator=gen)[: min(shots, len(idx))]])
    return rows.take(torch.cat(picked).numpy())

