"""Few-shot class-incremental experiment (Table 10).

The backbone and slot attention stay frozen (slots are read from DINOSAUR directly).
The router is meta-trained on the base session only. The support set starts with
``shot`` random images per base class and grows by the few-shot images of every new
session; test images are scored with the COMPOSE blend against it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from compose.cil.loop import CILConfig, accuracy, exemplars, meta_train
from compose.config import TrainConfig
from compose.engine.runner import set_seed
from compose.fscil.data import FSCILData
from compose.model import build_router
from compose.slots.dinosaur import DINOSAUR


@dataclass(frozen=True)
class FSCILRun:
    name: str
    train: TrainConfig = TrainConfig()


@dataclass(frozen=True)
class FSCILExperiment:
    name: str
    runs: tuple[FSCILRun, ...]
    datasets: tuple[str, ...] = ("cifar100", "cub200", "mini_imagenet")
    seeds: tuple[int, ...] = (42, 123, 7)
    # Router meta-training on the base session; no update in later sessions.
    meta: CILConfig = field(default_factory=lambda: CILConfig(episodes_per_session=0))


def load_fscil_experiment(path: str | Path) -> FSCILExperiment:
    with open(path) as f:
        raw = yaml.safe_load(f)
    known = {f.name for f in fields(CILConfig)}
    meta = {"episodes_per_session": 0, **raw.pop("meta", {})}
    if set(meta) - known:
        raise ValueError(f"unknown meta keys: {sorted(set(meta) - known)}")
    runs = tuple(FSCILRun(r["name"], TrainConfig(**r.get("train", {}))) for r in raw.pop("runs"))
    return FSCILExperiment(name=raw.pop("name"), runs=runs, meta=CILConfig(**meta),
                           **{k: tuple(v) for k, v in raw.items()})


@torch.no_grad()
def extract(dataset, dinosaur: DINOSAUR, *, device, generator, batch_size: int = 64,
            num_workers: int = 4, desc: str = "") -> tuple[torch.Tensor, torch.Tensor]:
    """``phi_full = [L2norm(CLS), phi_tilde_1..K]`` from DINOSAUR, in dataset order."""
    phis, labels = [], []
    for images, y in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
                          desc=desc, leave=False):
        out = dinosaur(images.to(device), generator=generator)
        phis.append(torch.cat([F.normalize(out["cls"], dim=-1).unsqueeze(1), out["phi_tilde"]], dim=1).cpu())
        labels.append(torch.as_tensor(y).long())
    return torch.cat(phis), torch.cat(labels)


def run_fscil(model, store: dict, n_sessions: int, shot: int, cfg: CILConfig, device, seed: int) -> dict:
    """Accuracies in percent: per session, ``A_T`` (final), ``AA`` (mean), ``FF`` (base-class drop)."""
    sup_phi, sup_lbl = exemplars(store["base_train_phi"], store["base_train_lbl"], shot, seed)
    base_test = store["test_0_phi"], store["test_0_lbl"]
    accs, base_accs = [], []
    for s in range(n_sessions):
        if s > 0:
            sup_phi = torch.cat([sup_phi, store[f"train_{s}_phi"]])
            sup_lbl = torch.cat([sup_lbl, store[f"train_{s}_lbl"]])
        accs.append(accuracy(model, sup_phi, sup_lbl, store[f"test_{s}_phi"], store[f"test_{s}_lbl"], cfg))
        base_accs.append(accuracy(model, sup_phi, sup_lbl, *base_test, cfg))
        print(f"    session {s + 1:02d}/{n_sessions}  acc={100 * accs[-1]:.2f}  "
              f"base acc={100 * base_accs[-1]:.2f}", flush=True)
    return {
        "A_T": 100.0 * accs[-1],
        "AA": 100.0 * float(np.mean(accs)),
        "FF": 100.0 * (base_accs[0] - base_accs[-1]),
        "session_acc": [100.0 * a for a in accs],
        "base_session_acc": [100.0 * a for a in base_accs],
    }


def run_fscil_experiment(exp: FSCILExperiment, *, roots: dict[str, str | None], index_dir: str,
                         dinosaur_ckpt: str, out_dir: str | Path, device: torch.device, extract_seed: int = 0) -> None:
    out_dir = Path(out_dir) / exp.name
    dinosaur = None
    for dataset in exp.datasets:
        folder = out_dir / dataset
        folder.mkdir(parents=True, exist_ok=True)
        todo = [(r, s) for r in exp.runs for s in exp.seeds if not (folder / f"{r.name}_seed{s}.json").exists()]
        if not todo:
            continue
        feats_path = folder / "features.pt"
        data = FSCILData(dataset, roots.get(dataset), index_dir)
        n_sessions = data.spec.sessions
        if feats_path.exists():
            store = torch.load(feats_path)
        else:
            dinosaur = dinosaur or DINOSAUR.from_checkpoint(dinosaur_ckpt).to(device)
            gen = torch.Generator().manual_seed(extract_seed)
            ex = lambda ds, d: extract(ds, dinosaur, device=device, generator=gen, desc=f"{dataset} {d}")
            store = {}
            store["base_train_phi"], store["base_train_lbl"] = ex(data.base_train(), "base train")
            for s in range(n_sessions):
                if s > 0:
                    store[f"train_{s}_phi"], store[f"train_{s}_lbl"] = ex(data.session_train(s), f"session {s} train")
                store[f"test_{s}_phi"], store[f"test_{s}_lbl"] = ex(data.test_until(s), f"session {s} test")
            torch.save(store, feats_path)

        for run, seed in todo:
            print(f"\n=== {exp.name} | {dataset} | {run.name} | seed {seed} ===", flush=True)
            t0 = time.time()
            set_seed(seed)
            model = build_router(run.train, int(store["base_train_phi"].shape[-1]))
            model = meta_train(model, store["base_train_phi"], store["base_train_lbl"],
                               n_episodes=exp.meta.episodes_base, cfg=exp.meta, device=device)
            metrics = run_fscil(model, store, n_sessions, data.spec.shot, exp.meta, device, seed)
            with (folder / f"{run.name}_seed{seed}.json").open("w") as f:
                json.dump({"experiment": exp.name, "dataset": dataset, "run": run.name, "seed": seed,
                           "train": asdict(run.train), "meta": asdict(exp.meta), "metrics": metrics,
                           "minutes": (time.time() - t0) / 60}, f, indent=2)
            print(f"  A_T={metrics['A_T']:.2f}  AA={metrics['AA']:.2f}  FF={metrics['FF']:.2f}", flush=True)
