"""Aggregate result files over seeds into the paper's table format.

    python scripts/summarize.py results/main --shot 10
    python scripts/summarize.py results/main --shot 10 --metrics Anoc Ha --variants COMPOSE

Each cell is the mean over seeds; ``±`` is the 95% Student-t interval half-width,
as in the tables of the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compose.metrics import across_seeds  # noqa: E402

MODES = {"cgqa": ["Asys", "Apro", "Asub", "Anon", "Anoc", "Ha"], "cobj": ["Asys", "Apro", "Anon", "Anoc", "Ha"]}
CONTINUAL = ["Acon", "AIA", "FF"]


def _value(summary: dict, key: str) -> float | None:
    v = summary.get(key)
    return v[0] if isinstance(v, list) else v


def collect(folder: Path, shot: str) -> dict:
    """``{(dataset, run, variant): {metric: [value per seed]}}``."""
    cells: dict = defaultdict(lambda: defaultdict(list))
    for path in sorted(folder.glob("*/*.json")):
        res = json.loads(path.read_text())
        ds, run = res["dataset"], res["run"]
        for key in CONTINUAL:
            cells[(ds, run, "(continual)")][key].append(res["continual"][key])
        for variant, summary in res["fewshot"].get(shot, {}).items():
            for key in summary:
                value = _value(summary, key)
                if value is not None:
                    cells[(ds, run, variant)][key].append(value)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="results/<experiment>")
    ap.add_argument("--shot", default="10")
    ap.add_argument("--metrics", nargs="*")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--continual", action="store_true", help="also print Phase I metrics")
    args = ap.parse_args()

    cells = collect(args.folder, args.shot)
    for ds in sorted({k[0] for k in cells}):
        metrics = args.metrics or MODES.get(ds, [])
        print(f"\n{ds.upper()}  ({args.shot}-shot)")
        print(f"  {'run':<16}{'variant':<20}{'n':>2}  " + "".join(f"{m:>15}" for m in metrics))
        for (d, run, variant), values in sorted(cells.items()):
            if d != ds or variant == "(continual)" or (args.variants and variant not in args.variants):
                continue
            row = []
            for m in metrics:
                s = across_seeds(values[m]) if values.get(m) else None
                row.append(f"{s['mean']:8.2f} ±{s['ci95']:4.2f}" if s else f"{'-':>15}")
            print(f"  {run:<16}{variant:<20}{len(values[metrics[0]]):>2}  " + "".join(f"{c:>15}" for c in row))
        if args.continual:
            for (d, run, variant), values in sorted(cells.items()):
                if d == ds and variant == "(continual)":
                    stats = {m: across_seeds(values[m]) for m in CONTINUAL}
                    print(f"  {run:<16}continual             " + "  ".join(
                        f"{m}={stats[m]['mean']:.2f}±{stats[m]['ci95']:.2f}" for m in CONTINUAL))


if __name__ == "__main__":
    main()
