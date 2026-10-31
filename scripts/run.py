"""Train and evaluate COMPOSE on CFST from pre-extracted slots.

    python scripts/run.py configs/cfst/main.yaml --cfst-root data/CFST --slots data/slots
    python scripts/run.py configs/cfst/main.yaml ... --datasets cgqa --runs COMPOSE --seeds 42

``--slots`` holds ``cgqa_slots_K7.pt`` / ``cobj_slots_K7.pt`` (``scripts/extract_slots.py``), and
``<dataset>_slots_K7_<source>.pt`` for runs that set ``slots: <source>`` (other backbones).
Results go to ``<out>/<experiment>/<dataset>/<run>_seed<seed>.{json,pt}``; finished
files are skipped, so the same command resumes an interrupted experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compose.config import load_experiment  # noqa: E402
from compose.engine.runner import default_device, run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--cfst-root", required=True, help="folder containing CGQA/ and COBJ/")
    ap.add_argument("--slots", required=True, help="folder with <dataset>_slots_K7.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--no-weights", action="store_true", help="do not save router weights")
    args = ap.parse_args()

    exp = load_experiment(args.config)
    run_experiment(
        exp,
        cfst_root=args.cfst_root,
        stores=lambda ds, src: str(Path(args.slots) / (f"{ds}_slots_K7.pt" if src is None else f"{ds}_slots_K7_{src}.pt")),
        out_dir=args.out,
        device=torch.device(args.device) if args.device else default_device(),
        runs=args.runs,
        datasets=args.datasets,
        seeds=args.seeds,
        save_weights=not args.no_weights,
    )


if __name__ == "__main__":
    main()
