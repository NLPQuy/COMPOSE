"""Experiment configuration. Defaults are the settings of every COMPOSE result in the paper."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from compose.model.inference import InferenceConfig


@dataclass(frozen=True)
class TrainConfig:
    """Phase I: one router per (dataset, seed)."""

    router: str = "compose"          # compose | compose_ct | bigru
    holistic_weight: float = 1.0     # compose_ct only: weight of the holistic logits in the loss
    slot_source: str = "phi_tilde"   # phi_tilde | gru_state
    hidden_dim: int = 64             # h
    lambda_d: float = 0.02           # cross-correlation weight
    kappa: int = 4                   # top slots kept for matching
    episodes_per_session: int = 300
    lr: float = 1e-3
    n_way: int = 10
    n_support: int = 10
    n_query: int = 10
    exemplars_per_class: int = 20
    eval_gamma: float = 0.3          # blend used to score the continual test sets
    eval_batch: int = 256            # query batch size; affects speed only


@dataclass(frozen=True)
class EvalConfig:
    """Phase II: CFST few-shot protocol (10-way, 10 queries, 300 episodes per mode)."""

    n_way: int = 10
    n_query: int = 10
    n_episodes: int = 300
    shots: tuple[int, ...] = (10,)


@dataclass(frozen=True)
class Run:
    """One training configuration, read by one or more inference variants."""

    name: str
    train: TrainConfig = TrainConfig()
    variants: tuple[InferenceConfig, ...] = (InferenceConfig(),)
    slots: str | None = None        # patch source of the slot store (None = DINOv2, the default store)


@dataclass(frozen=True)
class Experiment:
    name: str
    runs: tuple[Run, ...]
    datasets: tuple[str, ...] = ("cgqa", "cobj")
    seeds: tuple[int, ...] = (42, 123, 7)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _build(cls, raw: dict | None):
    raw = dict(raw or {})
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    for f in fields(cls):
        if f.name in raw and isinstance(raw[f.name], list):
            raw[f.name] = tuple(raw[f.name])
    return cls(**raw)


def load_experiment(path: str | Path) -> Experiment:
    """Read an experiment from YAML (see ``configs/``)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    defaults = raw.pop("train", {})               # shared by every run, overridable per run
    default_variants = raw.pop("variants", [{}])  # likewise
    runs = []
    for r in raw.pop("runs"):
        train = _build(TrainConfig, {**defaults, **r.get("train", {})})
        kappa = train.kappa
        variants = tuple(
            _build(InferenceConfig, {"kappa": kappa, **v}) for v in r.get("variants", default_variants)
        )
        runs.append(Run(name=r["name"], train=train, variants=variants, slots=r.get("slots")))
    return Experiment(
        name=raw.pop("name"),
        runs=tuple(runs),
        eval=_build(EvalConfig, raw.pop("eval", {})),
        **{k: tuple(v) for k, v in raw.items()},
    )
