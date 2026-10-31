"""Run an :class:`~compose.config.Experiment`: train each run once per (dataset, seed),
then evaluate all its inference variants at every shot count."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from compose.config import Experiment
from compose.data.slot_store import SlotSource, SlotStore
from compose.engine.continual import run_continual
from compose.engine.fewshot import run_fewshot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_experiment(
    exp: Experiment,
    *,
    cfst_root: str,
    stores,
    out_dir: str | Path,
    device: torch.device | None = None,
    runs: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    save_weights: bool = True,
) -> None:
    """Write ``<out_dir>/<dataset>/<run>_seed<seed>.json`` (and ``.pt`` router weights).

    ``stores`` maps a dataset to its slot store, or is a function ``(dataset, patch source)``
    returning the store path (runs with ``slots:`` read another backbone's store).

    Existing result files are skipped, so an interrupted experiment can be resumed.
    """
    device = device or default_device()
    out_dir = Path(out_dir) / exp.name
    loaded: dict[str, SlotStore] = {}
    views: dict[str, SlotStore] = {}       # one per (dataset, slot source), reused across runs

    for dataset in datasets or exp.datasets:
        for run in exp.runs:
            if runs and run.name not in runs:
                continue
            for seed in seeds or exp.seeds:
                path = out_dir / dataset / f"{run.name}_seed{seed}.json"
                if path.exists():
                    print(f"[skip] {path} exists")
                    continue
                file_key = f"{dataset}/{run.slots}"
                if file_key not in loaded:
                    store_path = stores(dataset, run.slots) if callable(stores) else stores[dataset]
                    loaded[file_key] = SlotStore.from_file(store_path)
                key = f"{file_key}/{run.train.slot_source}"
                if key not in views:
                    views[key] = loaded[file_key].with_source(SlotSource(run.train.slot_source))
                store = views[key]

                print(f"\n=== {exp.name} | {dataset} | {run.name} | seed {seed} ===", flush=True)
                t0 = time.time()
                set_seed(seed)
                model, continual = run_continual(dataset, cfst_root, store, run.train, seed=seed, device=device)
                fewshot = {
                    str(n_shot): run_fewshot(
                        dataset, cfst_root, store, model, exp.eval, run.variants, n_shot=n_shot, seed=seed,
                    )
                    for n_shot in exp.eval.shots
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                if save_weights:
                    torch.save(model.state_dict(), path.with_suffix(".pt"))
                with path.open("w") as f:
                    json.dump({
                        "experiment": exp.name,
                        "dataset": dataset,
                        "run": run.name,
                        "seed": seed,
                        "train": asdict(run.train),
                        "eval": asdict(exp.eval),
                        "variants": [asdict(v) for v in run.variants],
                        "slots": run.slots or "dinov2_b14",
                        "slot_store": loaded[file_key].metadata,
                        "continual": continual,
                        "fewshot": fewshot,
                        "minutes": (time.time() - t0) / 60,
                    }, f, indent=2, default=str)
                print(f"saved {path}", flush=True)
