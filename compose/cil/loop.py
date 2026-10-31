"""Class-incremental loop of COMPOSE on standard benchmarks (Table 11).

The router is meta-trained on the base session and again on each new session. A replay
buffer keeps ``exemplars_per_class`` images of every class seen so far; test images are
scored with the COMPOSE blend (Eq. 9) against the buffer as support set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from compose.model.inference import InferenceConfig, predict
from compose.model.router import SlotRouter


@dataclass(frozen=True)
class CILConfig:
    episodes_base: int = 1000
    episodes_per_session: int = 200
    lr: float = 1e-3
    n_way: int = 10
    n_support: int = 5
    n_query: int = 10
    episode_seed: int = 42       # seed of the episode sampler, reset for every meta-training call
    exemplars_per_class: int = 20
    gamma: float = 0.3
    kappa: int = 4
    eval_batch: int = 256


def sample_episode(phi, labels, *, n_way, n_support, n_query, rng):
    """Episode from tensors; classes with too few images contribute what they have."""
    classes = labels.unique()
    n_way = min(n_way, len(classes))
    chosen = classes[torch.randperm(len(classes), generator=rng)[:n_way]].tolist()
    sup, qry, sup_lbl, qry_lbl = [], [], [], []
    for rel, c in enumerate(chosen):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(idx), generator=rng)
        s, q = idx[perm[:n_support]], idx[perm[n_support:n_support + n_query]]
        sup.append(phi[s]); qry.append(phi[q])
        sup_lbl += [rel] * len(s); qry_lbl += [rel] * len(q)
    return torch.cat(sup), torch.tensor(sup_lbl), torch.cat(qry), torch.tensor(qry_lbl)


def meta_train(model: SlotRouter, phi, labels, *, n_episodes: int, cfg: CILConfig, device) -> SlotRouter:
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes)
    rng = torch.Generator()
    rng.manual_seed(cfg.episode_seed)
    for _ in range(n_episodes):
        sup, sup_lbl, qry, qry_lbl = sample_episode(
            phi, labels, n_way=cfg.n_way, n_support=cfg.n_support, n_query=cfg.n_query, rng=rng)
        sup, sup_lbl, qry, qry_lbl = sup.to(device), sup_lbl.to(device), qry.to(device), qry_lbl.to(device)
        logits, reg = model.episode_logits(sup, sup_lbl, qry)
        loss = F.cross_entropy(logits, qry_lbl)
        if reg.item() > 0:
            loss = loss + reg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
    return model.eval()


def exemplars(phi, labels, shots: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Up to ``shots`` random images per class (the replay buffer entries of one session)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    out_phi, out_lbl = [], []
    for c in labels.unique().tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        sel = idx[torch.randperm(len(idx), generator=rng)[:shots]]
        out_phi.append(phi[sel])
        out_lbl.append(torch.full((len(sel),), c, dtype=torch.long))
    return torch.cat(out_phi), torch.cat(out_lbl)


@torch.no_grad()
def accuracy(model, sup_phi, sup_lbl, phi, labels, cfg: CILConfig) -> float:
    infer = InferenceConfig(gamma=cfg.gamma, kappa=cfg.kappa)
    correct = 0
    for i in range(0, len(phi), cfg.eval_batch):
        pred = predict(model, sup_phi, sup_lbl, phi[i:i + cfg.eval_batch], infer)
        correct += int((pred == labels[i:i + cfg.eval_batch].cpu()).sum())
    return correct / len(phi)


def run_cil(model: SlotRouter, sessions: list[dict], cfg: CILConfig, device, seed: int) -> dict:
    """``sessions[t]`` holds ``train_phi/train_lbl`` (new classes) and ``test_phi/test_lbl`` (all seen).

    Returns accuracies in percent: ``A_T`` (final session), ``AA`` (mean over sessions),
    ``FF`` (drop of base-session accuracy from the first to the last session).
    """
    buf_phi = buf_lbl = None
    accs, base_accs = [], []
    for t, s in enumerate(sessions):
        n_ep = cfg.episodes_base if t == 0 else cfg.episodes_per_session
        if n_ep > 0:
            model = meta_train(model, s["train_phi"], s["train_lbl"], n_episodes=n_ep, cfg=cfg, device=device)
        new_phi, new_lbl = exemplars(s["train_phi"], s["train_lbl"], cfg.exemplars_per_class, seed)
        buf_phi = new_phi if buf_phi is None else torch.cat([buf_phi, new_phi])
        buf_lbl = new_lbl if buf_lbl is None else torch.cat([buf_lbl, new_lbl])
        accs.append(accuracy(model, buf_phi, buf_lbl, s["test_phi"], s["test_lbl"], cfg))
        base_accs.append(accuracy(model, buf_phi, buf_lbl, sessions[0]["test_phi"], sessions[0]["test_lbl"], cfg))
        print(f"    session {t + 1:02d}/{len(sessions)}  acc={100 * accs[-1]:.2f}  "
              f"base-session acc={100 * base_accs[-1]:.2f}", flush=True)
    return {
        "A_T": 100.0 * accs[-1],
        "AA": 100.0 * float(np.mean(accs)),
        "FF": 100.0 * (base_accs[0] - base_accs[-1]),
        "session_acc": [100.0 * a for a in accs],
        "base_session_acc": [100.0 * a for a in base_accs],
    }
