"""Standard class-incremental experiment (Table 11): adapt, extract, then run COMPOSE."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import yaml

from compose.cil.adapter import AdapterViT
from compose.cil.base_session import BaseSessionConfig, adapt_backbone
from compose.cil.data import CILData
from compose.cil.features import extract
from compose.cil.loop import CILConfig, run_cil
from compose.config import TrainConfig
from compose.engine.runner import set_seed
from compose.model import build_router
from compose.slots.dinosaur import DINOSAUR


@dataclass(frozen=True)
class CILRun:
    name: str
    train: TrainConfig = TrainConfig()


@dataclass(frozen=True)
class CILExperiment:
    name: str
    runs: tuple[CILRun, ...]
    datasets: tuple[str, ...] = ("cifar224", "imagenetr", "imageneta")
    seeds: tuple[int, ...] = (42,)
    base_session: BaseSessionConfig = field(default_factory=BaseSessionConfig)
    cil: CILConfig = field(default_factory=CILConfig)


def _build(cls, raw):
    known = {f.name for f in fields(cls)}
    if set(raw or {}) - known:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(set(raw) - known)}")
    return cls(**(raw or {}))


def load_cil_experiment(path: str | Path) -> CILExperiment:
    with open(path) as f:
        raw = yaml.safe_load(f)
    runs = tuple(CILRun(r["name"], _build(TrainConfig, r.get("train"))) for r in raw.pop("runs"))
    return CILExperiment(
        name=raw.pop("name"), runs=runs,
        base_session=_build(BaseSessionConfig, raw.pop("base_session", None)),
        cil=_build(CILConfig, raw.pop("cil", None)),
        **{k: tuple(v) for k, v in raw.items()},
    )


def _sessions(data: CILData, train, test) -> list[dict]:
    (tr_phi, tr_lbl), (te_phi, te_lbl) = train, test
    out = []
    for r in data.session_ranges():
        new = (tr_lbl >= r.start) & (tr_lbl < r.stop)
        seen = te_lbl < r.stop
        out.append({"train_phi": tr_phi[new], "train_lbl": tr_lbl[new],
                    "test_phi": te_phi[seen], "test_lbl": te_lbl[seen]})
    return out


def run_cil_experiment(exp: CILExperiment, *, roots: dict[str, str], dinosaur_ckpt: str, out_dir: str | Path,
                       device: torch.device) -> None:
    """Write ``<out>/<experiment>/<dataset>/<run>_seed<seed>.json``.

    The adapted backbone (adapter weights) and the extracted features are cached next
    to the results, so changing only the COMPOSE runs does not repeat those steps.
    """
    out_dir = Path(out_dir) / exp.name
    dinosaur = DINOSAUR.from_checkpoint(dinosaur_ckpt).to(device)
    for dataset in exp.datasets:
        data = CILData(dataset, roots[dataset])
        for seed in exp.seeds:
            folder = out_dir / dataset
            folder.mkdir(parents=True, exist_ok=True)
            todo = [r for r in exp.runs if not (folder / f"{r.name}_seed{seed}.json").exists()]
            if not todo:
                continue
            feats_path = folder / f"features_seed{seed}.pt"
            if feats_path.exists():
                feats = torch.load(feats_path)
            else:
                adapter_path = folder / f"adapter_seed{seed}.pt"
                set_seed(seed)
                if adapter_path.exists():
                    vit = AdapterViT(exp.base_session.ffn_num)
                    vit.load_state_dict(torch.load(adapter_path))
                    vit = vit.to(device).eval()
                else:
                    print(f"\n=== {dataset} seed {seed}: base-session adaptation ===", flush=True)
                    base = data.session_ranges()[0]
                    vit = adapt_backbone(data.subset(base, "train", train_transform=True), len(base),
                                         exp.base_session, device)
                    torch.save(vit.state_dict(), adapter_path)
                gen = torch.Generator().manual_seed(seed)
                all_classes = range(data.spec.num_classes)
                feats = {
                    "train": extract(data.subset(all_classes, "train"), vit, dinosaur, device=device,
                                     generator=gen, desc=f"{dataset} train"),
                    "test": extract(data.subset(all_classes, "test"), vit, dinosaur, device=device,
                                    generator=gen, desc=f"{dataset} test"),
                }
                torch.save(feats, feats_path)
            sessions = _sessions(data, feats["train"], feats["test"])

            for run in todo:
                print(f"\n=== {exp.name} | {dataset} | {run.name} | seed {seed} ===", flush=True)
                t0 = time.time()
                set_seed(seed)
                model = build_router(run.train, int(feats["train"][0].shape[-1]))
                metrics = run_cil(model, sessions, exp.cil, device, seed)
                with (folder / f"{run.name}_seed{seed}.json").open("w") as f:
                    json.dump({"experiment": exp.name, "dataset": dataset, "run": run.name, "seed": seed,
                               "train": asdict(run.train), "cil": asdict(exp.cil),
                               "base_session": asdict(exp.base_session), "metrics": metrics,
                               "minutes": (time.time() - t0) / 60}, f, indent=2)
                print(f"  A_T={metrics['A_T']:.2f}  AA={metrics['AA']:.2f}  FF={metrics['FF']:.2f}", flush=True)
