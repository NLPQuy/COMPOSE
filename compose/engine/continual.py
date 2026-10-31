"""Phase I: class-incremental training over the CFST continual sessions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from compose.config import TrainConfig
from compose.data import cfst
from compose.data.sampling import Rows, continual_task_splits, task_rows
from compose.data.slot_store import SlotStore
from compose.engine.train import exemplar_rows, meta_train
from compose.metrics import continual_metrics
from compose.model import build_router
from compose.model.inference import InferenceConfig, predict
from compose.model.router import SlotRouter


@torch.no_grad()
def accuracy(
    model: SlotRouter,
    store: SlotStore,
    sup_phi: torch.Tensor,
    sup_lbl: torch.Tensor,
    test: Sequence[Rows],
    cfg: InferenceConfig,
    batch_size: int,
) -> float:
    """Accuracy (%) of ``test`` against the exemplar support set, in query batches."""
    correct = total = 0
    for rows in test:
        for start in range(0, len(rows), batch_size):
            chunk = rows.take(torch.arange(start, min(start + batch_size, len(rows))).numpy())
            pred = predict(model, sup_phi, sup_lbl, store.gather(chunk), cfg)
            correct += int((pred == torch.as_tensor(chunk.labels)).sum().item())
            total += len(chunk)
    return 100.0 * correct / max(1, total)


def run_continual(
    dataset: str,
    cfst_root: str,
    store: SlotStore,
    cfg: TrainConfig,
    *,
    seed: int,
    device: torch.device,
) -> tuple[SlotRouter, dict]:
    """Train the router session by session; return it and the continual metrics."""
    spec = cfst.get_spec(dataset)
    targets = {s: cfst.split_targets(cfst_root, dataset, f"continual_{s}") for s in ("train", "test")}
    for s, t in targets.items():
        store.check(f"continual_{s}", t)
    per_task, mappings = continual_task_splits(targets["train"], spec.num_tasks, seed)
    train = task_rows("continual_train", targets["train"], per_task, mappings)
    test = task_rows("continual_test", targets["test"], per_task, mappings)

    model = build_router(cfg, store.feature_dim)
    eval_cfg = InferenceConfig(gamma=cfg.eval_gamma, kappa=cfg.kappa)
    bank: list[Rows] = []
    acc_matrix, joint = [], []

    for t, session in enumerate(train):
        print(f"[{dataset}] session {t + 1}/{spec.num_tasks}", flush=True)
        model = meta_train(
            model, store, session,
            n_way=min(cfg.n_way, spec.way_per_task),
            n_support=cfg.n_support,
            n_query=cfg.n_query,
            n_episodes=cfg.episodes_per_session,
            lr=cfg.lr,
            seed=seed + t,
            device=device,
        )
        bank.append(exemplar_rows(session, cfg.exemplars_per_class, seed + 1000 + t))
        sup_phi = torch.cat([store.gather(r) for r in bank])
        sup_lbl = torch.cat([torch.as_tensor(r.labels, dtype=torch.long) for r in bank])

        joint.append(accuracy(model, store, sup_phi, sup_lbl, test[: t + 1], eval_cfg, cfg.eval_batch))
        acc_matrix.append([
            accuracy(model, store, sup_phi, sup_lbl, [test[i]], eval_cfg, cfg.eval_batch)
            for i in range(t + 1)
        ])
        print(f"    joint={joint[-1]:.2f}  per-session={[round(x, 2) for x in acc_matrix[-1]]}", flush=True)

    return model.eval(), continual_metrics(acc_matrix, joint)
