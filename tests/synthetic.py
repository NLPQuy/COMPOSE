"""A tiny synthetic CFST root (JSON only, no images) and a matching slot store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compose.data import cfst

D, K, D_GRU = 16, 7, 8


def _write(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items))


def make_cfst(root: Path, per_fewshot_class: int = 6) -> Path:
    rng = np.random.RandomState(0)
    base = root / "CGQA" / "GQA_100"
    combos = [[f"o{c // 10}", f"a{c % 10}"] for c in range(100)]
    for split, per in (("train", 8), ("val", 1), ("test", 3)):
        items = [{"newImageName": f"continual/{split}/{i}", "comb": combos[c]}
                 for i, c in enumerate(rng.permutation(np.repeat(np.arange(100), per)))]
        _write(base / "continual" / split / f"{split}.json", items)
    for mode, rel in (("sys", "sys/sys_fewshot.json"), ("pro", "pro/pro_fewshot.json"),
                      ("sub", "sub/sub_fewshot.json"), ("non", "non_novel/non_novel_fewshot.json"),
                      ("noc", "non_comp/non_comp_fewshot.json")):
        items = [{"newImageName": f"fewshot/{mode}/{i}", "comb": [f"{mode}{c}", "x"]}
                 for i, c in enumerate(rng.permutation(np.repeat(np.arange(12), per_fewshot_class)))]
        _write(base / "fewshot" / rel, items)
    ann = root / "COBJ" / "annotations"
    labels = [[f"p{c // 5}", f"q{c % 5}"] for c in range(30)]
    for split, per in (("train", 8), ("val", 1), ("test", 3)):
        items = [{"imageId": i, "label": labels[c]}
                 for i, c in enumerate(rng.permutation(np.repeat(np.arange(30), per)))]
        _write(ann / f"O365_continual_{split}_crop.json", items)
    for mode in ("sys", "pro", "non", "noc"):
        items = [{"imageId": i, "label": [f"{mode}{c}", "y"]}
                 for i, c in enumerate(rng.permutation(np.repeat(np.arange(12), per_fewshot_class)))]
        _write(ann / f"O365_{mode}_fewshot_crop.json", items)
    return root


def make_store(cfst_root: Path, dataset: str, path: Path) -> Path:
    """Slots clustered by class, so a trained router should beat chance."""
    g = torch.Generator().manual_seed(1)
    out = {}
    for split in cfst.split_names(dataset):
        y = torch.as_tensor(cfst.split_targets(cfst_root, dataset, split))
        n = len(y)
        centers = torch.randn(int(y.max()) + 1, K, D, generator=g)
        phi = centers[y] + 0.8 * torch.randn(n, K, D, generator=g)
        out.update({
            f"{split}_phi": phi,
            f"{split}_phi_tilde": F.normalize(phi, dim=-1),
            f"{split}_cls": phi.mean(1),
            f"{split}_slots": torch.randn(n, K, D_GRU, generator=g),
            f"{split}_labels": y,
        })
    out["metadata"] = {"dataset": dataset, "n_slots": K, "phi_dim": D, "ckpt": "synthetic"}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return path


SMALL_TRAIN = dict(episodes_per_session=6, n_way=3, n_support=2, n_query=2, exemplars_per_class=3, eval_batch=7)
SMALL_EVAL = dict(n_way=3, n_query=2, n_episodes=4, shots=(2,))
