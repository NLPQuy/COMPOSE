import json

import torch

from compose.config import EvalConfig, Experiment, Run, TrainConfig, load_experiment
from compose.engine.runner import run_experiment
from compose.model.inference import InferenceConfig
from tests.synthetic import SMALL_EVAL, SMALL_TRAIN

CPU = torch.device("cpu")


def _experiment(name):
    variants = (InferenceConfig("COMPOSE", 0.3), InferenceConfig("w/o Chamfer", 1.0))
    runs = (
        Run("COMPOSE", TrainConfig(**SMALL_TRAIN), variants),
        Run("COMPOSE-CT", TrainConfig(**SMALL_TRAIN, router="compose_ct", holistic_weight=0.3)),
        Run("GRU-state", TrainConfig(**SMALL_TRAIN, slot_source="gru_state")),
    )
    return Experiment(name, runs, seeds=(42,), eval=EvalConfig(**SMALL_EVAL))


def test_runs_are_deterministic_and_resumable(data, tmp_path, capsys):
    cfst_root, stores = data
    for out in ("a", "b"):
        run_experiment(_experiment("e2e"), cfst_root=str(cfst_root), stores=stores, out_dir=tmp_path / out, device=CPU)
    for ds in ("cgqa", "cobj"):
        for run in ("COMPOSE", "COMPOSE-CT", "GRU-state"):
            a = json.loads((tmp_path / "a" / "e2e" / ds / f"{run}_seed42.json").read_text())
            b = json.loads((tmp_path / "b" / "e2e" / ds / f"{run}_seed42.json").read_text())
            assert a["continual"] == b["continual"] and a["fewshot"] == b["fewshot"]
            assert 0 <= a["fewshot"]["2"]["COMPOSE"]["Ha"] <= 100
    assert "Asub" not in a["fewshot"]["2"]["COMPOSE"]           # COBJ has no sub mode
    capsys.readouterr()
    run_experiment(_experiment("e2e"), cfst_root=str(cfst_root), stores=stores, out_dir=tmp_path / "a", device=CPU)
    assert capsys.readouterr().out.count("[skip]") == 6


def test_shipped_configs_load():
    from pathlib import Path

    for path in sorted((Path(__file__).parents[1] / "configs" / "cfst").glob("*.yaml")):
        exp = load_experiment(path)
        assert exp.runs and all(r.variants for r in exp.runs), path
