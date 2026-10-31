"""Evaluation metrics (percent).

Continual (Phase I), from the lower-triangular accuracy matrix ``R[j][i]`` (accuracy
on task ``i`` after learning task ``j``) and the joint class-IL curve ``A_j``:

* ``Acon = A_{T-1}``, the final joint accuracy;
* ``AIA = mean_j A_j``;
* ``FF = mean_{i<T-1} (max_j R[j][i] - R[T-1][i])`` and ``BWT = -FF``.

Few-shot (Phase II, CFST): ``A_m`` is the mean episode accuracy of mode ``m`` with
``1.96 * std / sqrt(n)``; ``Hn``/``Hr``/``Ha`` are harmonic means over the available
novel (sys, pro, sub), reference (non, noc) and all modes. COBJ has no ``sub``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

HN_MODES = ("sys", "pro", "sub")
HR_MODES = ("non", "noc")
HA_MODES = HN_MODES + HR_MODES


def continual_metrics(acc_matrix: Sequence[Sequence[float]], joint: Sequence[float]) -> dict:
    rows = [[float(x) for x in row] for row in acc_matrix]
    joint = [float(x) for x in joint]
    if len(rows) != len(joint):
        raise ValueError("acc_matrix and joint curve must have one entry per task")
    t = len(rows)
    forgetting = [max(rows[j][i] for j in range(i, t)) - rows[-1][i] for i in range(t - 1)]
    ff = float(np.mean(forgetting)) if forgetting else 0.0
    return {
        "Acon": joint[-1],
        "AIA": float(np.mean(joint)),
        "FF": ff,
        "BWT": -ff,
        "acc_matrix": rows,
        "per_step_joint": joint,
    }


def mean_ci95(episode_accs: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(episode_accs, dtype=float)
    return float(arr.mean()), float(1.96 * arr.std() / math.sqrt(arr.size))


def harmonic_mean(values: Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if np.any(arr == 0):
        return 0.0
    return float(arr.size / np.sum(1.0 / arr))


def fewshot_summary(accs_by_mode: Mapping[str, Sequence[float]]) -> dict:
    """``{A<mode>: [mean, ci95], Hn, Hr, Ha}`` from per-episode accuracies."""
    out: dict = {}
    means = {}
    for mode, accs in accs_by_mode.items():
        mean, ci = mean_ci95(accs)
        out[f"A{mode}"] = [mean, ci]
        means[mode] = mean
    for key, modes in (("Hn", HN_MODES), ("Hr", HR_MODES), ("Ha", HA_MODES)):
        out[key] = harmonic_mean([means[m] for m in modes if m in means])
    return out


# Two-sided 97.5% quantile of Student's t, for the across-seed intervals of the paper tables.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def across_seeds(values: Sequence[float]) -> dict:
    """Mean, sample std and 95% t-interval half-width over seeds (the "±ci" of the tables)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci = _T975.get(n - 1, 1.96) * std / math.sqrt(n) if n > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "ci95": ci, "n": n}
