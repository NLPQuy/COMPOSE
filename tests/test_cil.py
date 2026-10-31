from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from compose.cil.data import SPECS
from compose.cil.loop import CILConfig, run_cil
from compose.model import ChamferTrainedRouter, SlotRouter

CPU = torch.device("cpu")


def _sessions(num_classes=30, init=10, inc=10, k=7, d=16):
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(num_classes, k + 1, d, generator=g)

    def split(per):
        y = torch.arange(num_classes).repeat_interleave(per)
        return F.normalize(centers[y] + 0.5 * torch.randn(len(y), k + 1, d, generator=g), dim=-1), y

    (tr, tr_y), (te, te_y) = split(10), split(4)
    out = []
    for stop in range(init, num_classes + 1, inc):
        new, seen = (tr_y >= stop - inc) & (tr_y < stop), te_y < stop
        out.append({"train_phi": tr[new], "train_lbl": tr_y[new], "test_phi": te[seen], "test_lbl": te_y[seen]})
    return out


@pytest.mark.parametrize("router", [SlotRouter, ChamferTrainedRouter])
def test_cil_loop_is_deterministic(router):
    cfg = CILConfig(episodes_base=10, episodes_per_session=3, exemplars_per_class=4)
    runs = []
    for _ in range(2):
        torch.manual_seed(42)
        runs.append(run_cil(router(16, hidden_dim=8), _sessions(), cfg, CPU, seed=42))
    assert runs[0] == runs[1]
    m = runs[0]
    assert len(m["session_acc"]) == 3 and m["A_T"] == m["session_acc"][-1]
    assert m["session_acc"][0] > 50                      # well-separated synthetic classes


def test_session_splits():
    assert SPECS["cifar224"].sessions == [10] * 10
    assert SPECS["imagenetr"].sessions == [20] * 10


def test_features_shapes():
    pytest.importorskip("timm")
    from compose.cil.adapter import AdapterViT
    from compose.cil.features import extract
    from compose.slots.dinosaur import DINOSAUR

    class Two(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return i, torch.rand(3, 224, 224), i

    vit = AdapterViT(ffn_num=4, dim=32, depth=1, num_heads=2).eval()
    phi, y = extract(Two(), vit, DINOSAUR().eval(), device=CPU, generator=torch.Generator().manual_seed(0),
                     num_workers=0)
    assert phi.shape == (2, 8, 32) and y.tolist() == [0, 1]
    assert torch.allclose(phi.norm(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_cil_config_loads():
    from compose.cil.runner import load_cil_experiment

    exp = load_cil_experiment(Path(__file__).parents[1] / "configs" / "cil" / "vit_cil.yaml")
    assert [r.name for r in exp.runs] == ["COMPOSE", "COMPOSE-CT"] and exp.base_session.epochs == 20


def test_fscil_loop_and_config():
    from compose.fscil.data import SPECS as FS
    from compose.fscil.runner import load_fscil_experiment, run_fscil

    assert [FS[d].sessions for d in ("cifar100", "cub200", "mini_imagenet")] == [9, 11, 9]
    exp = load_fscil_experiment(Path(__file__).parents[1] / "configs" / "cil" / "fscil.yaml")
    assert exp.meta.episodes_per_session == 0 and [r.name for r in exp.runs] == ["COMPOSE", "COMPOSE-CT"]

    g = torch.Generator().manual_seed(0)
    centers = torch.randn(14, 8, 16, generator=g)
    phi = lambda y: F.normalize(centers[y] + 0.5 * torch.randn(len(y), 8, 16, generator=g), dim=-1)
    y0, yt0 = torch.arange(10).repeat_interleave(6), torch.arange(10).repeat_interleave(3)
    y1, yt1 = torch.arange(10, 14).repeat_interleave(2), torch.arange(14).repeat_interleave(3)
    store = {"base_train_phi": phi(y0), "base_train_lbl": y0, "test_0_phi": phi(yt0), "test_0_lbl": yt0,
             "train_1_phi": phi(y1), "train_1_lbl": y1, "test_1_phi": phi(yt1), "test_1_lbl": yt1}
    torch.manual_seed(0)
    m = run_fscil(SlotRouter(16, hidden_dim=8), store, 2, 2, CILConfig(), CPU, seed=0)
    assert len(m["session_acc"]) == 2 and m["A_T"] == m["session_acc"][-1]
