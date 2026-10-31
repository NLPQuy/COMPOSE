"""COMPOSE on standard class-incremental benchmarks (Table 11).

    python scripts/run_cil.py configs/cil/vit_cil.yaml \
        --root cifar224=data/cifar --root imagenetr=data/imagenet-r --root imageneta=data/imagenet-a \
        --checkpoint checkpoints/coco_dinosaur_base14_dinov2.ckpt

CIFAR-100 is downloaded into its root if missing; ImageNet-R / ImageNet-A roots hold
``train/`` and ``test/`` class folders (the splits used by LAMDA-PILOT). Each dataset and
seed runs: base-session adaptation -> slot features -> every COMPOSE run in the config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compose.cil.runner import load_cil_experiment, run_cil_experiment  # noqa: E402
from compose.engine.runner import default_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--root", action="append", default=[], metavar="DATASET=PATH", required=True)
    ap.add_argument("--checkpoint", required=True, help="DINOSAUR checkpoint (see README)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    exp = load_cil_experiment(args.config)
    roots = dict(r.split("=", 1) for r in args.root)
    missing = [d for d in exp.datasets if d not in roots]
    if missing:
        sys.exit(f"missing --root for {missing}")
    run_cil_experiment(exp, roots=roots, dinosaur_ckpt=args.checkpoint, out_dir=args.out,
                       device=torch.device(args.device) if args.device else default_device())


if __name__ == "__main__":
    main()
