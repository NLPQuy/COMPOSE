"""COMPOSE on few-shot class-incremental benchmarks (Table 10).

    git clone --depth 1 https://github.com/icoz69/CEC-CVPR2021 data/CEC
    python scripts/run_fscil.py configs/cil/fscil.yaml --index-dir data/CEC/data/index_list \
        --root cifar100=data/cifar --root cub200=data/cub \
        --checkpoint checkpoints/coco_dinosaur_base14_dinov2.ckpt

CIFAR-100 is downloaded into its root if missing; the CUB root holds ``CUB_200_2011/``;
miniImageNet is read from Hugging Face (``timm/mini-imagenet``, needs ``pip install datasets``).
Slots are extracted once per dataset and cached in the output folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compose.engine.runner import default_device  # noqa: E402
from compose.fscil.runner import load_fscil_experiment, run_fscil_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--index-dir", required=True, help="CEC data/index_list folder")
    ap.add_argument("--root", action="append", default=[], metavar="DATASET=PATH")
    ap.add_argument("--checkpoint", required=True, help="DINOSAUR checkpoint (see README)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    exp = load_fscil_experiment(args.config)
    roots = dict(r.split("=", 1) for r in args.root)
    missing = [d for d in exp.datasets if d != "mini_imagenet" and d not in roots]
    if missing:
        sys.exit(f"missing --root for {missing}")
    run_fscil_experiment(exp, roots=roots, index_dir=args.index_dir, dinosaur_ckpt=args.checkpoint,
                         out_dir=args.out, device=torch.device(args.device) if args.device else default_device())


if __name__ == "__main__":
    main()
