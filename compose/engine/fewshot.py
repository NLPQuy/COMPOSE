"""Phase II: CFST few-shot evaluation of a trained router."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from compose.config import EvalConfig
from compose.data import cfst
from compose.data.sampling import fewshot_episodes
from compose.data.slot_store import SlotStore
from compose.metrics import fewshot_summary
from compose.model.inference import InferenceConfig, predict
from compose.model.router import SlotRouter


@torch.no_grad()
def run_fewshot(
    dataset: str,
    cfst_root: str,
    store: SlotStore,
    model: SlotRouter,
    cfg: EvalConfig,
    variants: Sequence[InferenceConfig],
    *,
    n_shot: int,
    seed: int,
) -> dict[str, dict]:
    """Per-mode accuracy of every inference variant on the same episodes.

    Slots are gathered once per episode and read by each variant; no weights change.
    Returns ``{variant name: fewshot_summary}``.
    """
    accs = {v.name: {} for v in variants}
    for mode in cfst.get_spec(dataset).fewshot_modes:
        split = f"fewshot_{mode}"
        targets = cfst.split_targets(cfst_root, dataset, split)
        store.check(split, targets)
        episodes = fewshot_episodes(
            split, targets, n_way=cfg.n_way, n_shot=n_shot, n_query=cfg.n_query,
            n_episodes=cfg.n_episodes, seed=seed,
        )
        per_variant = {v.name: [] for v in variants}
        for ep in episodes:
            sup_phi, qry_phi = store.gather(ep.support), store.gather(ep.query)
            sup_lbl = torch.as_tensor(ep.support.labels, dtype=torch.long)
            qry_lbl = torch.as_tensor(ep.query.labels, dtype=torch.long)
            for v in variants:
                pred = predict(model, sup_phi, sup_lbl, qry_phi, v)
                per_variant[v.name].append(float((pred == qry_lbl).float().mean().item() * 100.0))
        for name, a in per_variant.items():
            accs[name][mode] = a
        print(f"  {n_shot}-shot {mode}: " + "  ".join(
            f"{n}={sum(a) / len(a):.2f}" for n, a in per_variant.items()), flush=True)
    return {name: fewshot_summary(by_mode) for name, by_mode in accs.items()}
