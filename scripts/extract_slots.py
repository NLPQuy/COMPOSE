"""Extract DINOSAUR slots for every CFST split into one slot store per dataset.

    python scripts/extract_slots.py --dataset cgqa --cfst-root data/CFST \
        --checkpoint checkpoints/coco_dinosaur_base14_dinov2.ckpt --out data/slots

Writes ``<out>/<dataset>_slots_K7.pt`` (schema in ``compose/data/slot_store.py``).
Slot attention starts from random slots; ``--seed`` fixes that noise. A GPU takes
about 15 minutes for CGQA.

Other patch sources (Tables 2, 7, 8, 23, 25) keep the DINOSAUR attention and swap the tokens:

    python scripts/extract_slots.py --dataset cgqa ... --patch-source configs/backbones/mae_vith14_L15.yaml \
        --source-checkpoint checkpoints/mae_pretrain_vit_huge.pth

which writes ``<out>/<dataset>_slots_K7_<source>.pt``. The /16 sources need a DINOSAUR trained on
DINOv1 ViT-B/16 as ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compose.data import cfst  # noqa: E402
from compose.slots.backbones import PatchSource, build_backbone, readout  # noqa: E402
from compose.slots.dinosaur import CHECKPOINT_SHA256, CHECKPOINT_URL, DINOSAUR  # noqa: E402

DINOSAUR_BACKBONES = {"b14": "vit_base_patch14_dinov2", "b16": "vit_base_patch16_224.dino"}


def transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class Images(Dataset):
    """Each image at 224 (DINOSAUR) and, if different, at the patch source's resolution."""

    def __init__(self, root: Path, files: list[tuple[str, int]], source_size: int = 224):
        self.root, self.files = root, files
        self.t224, self.tsrc = transform(224), (transform(source_size) if source_size != 224 else None)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, label = self.files[i]
        img = Image.open(self.root / path).convert("RGB")
        return self.t224(img), (self.tsrc(img) if self.tsrc else 0), label


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def extract_split(model, root, files, *, batch_size, device, generator, desc,
                  source: PatchSource | None = None, backbone=None) -> dict[str, torch.Tensor]:
    size = source.input_size if source else 224
    loader = DataLoader(Images(root, files, size), batch_size=batch_size, shuffle=False, num_workers=4)
    parts: dict[str, list[torch.Tensor]] = {}
    for images, images_src, labels in tqdm(loader, desc=desc):
        images = images.to(device)
        out = model(images, generator=generator)
        if source is not None:
            out = readout(source, backbone, out, images, images_src.to(device) if size != 224 else None)
        for k, v in out.items():
            parts.setdefault(k, []).append(v.cpu())
        parts.setdefault("labels", []).append(labels.long())
    return {k: torch.cat(v) for k, v in parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(cfst.SPECS))
    ap.add_argument("--cfst-root", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path, help=f"from {CHECKPOINT_URL}")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-slots", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patch-source", type=Path, help="configs/backbones/<source>.yaml (default: DINOv2 itself)")
    ap.add_argument("--source-checkpoint", help="local weights of the patch source, if not from timm")
    args = ap.parse_args()

    source = PatchSource.from_yaml(args.patch_source, args.source_checkpoint) if args.patch_source else None
    dino = source.dinosaur if source else "b14"
    if dino == "b14" and sha256(args.checkpoint) != CHECKPOINT_SHA256:
        sys.exit(f"{args.checkpoint} is not the released DINOSAUR checkpoint (SHA-256 mismatch)")
    model = DINOSAUR.from_checkpoint(args.checkpoint, n_slots=args.n_slots,
                                     backbone=DINOSAUR_BACKBONES[dino]).to(args.device)
    backbone = build_backbone(source).to(args.device) if source else None
    generator = torch.Generator().manual_seed(args.seed)
    root = cfst.image_root(args.cfst_root, args.dataset)

    store: dict = {}
    for split in cfst.split_names(args.dataset):
        files = cfst.split_files(args.cfst_root, args.dataset, split)
        part = extract_split(model, root, files, batch_size=args.batch_size, device=args.device,
                             generator=generator, desc=f"{args.dataset} {split}", source=source, backbone=backbone)
        store.update({f"{split}_{k}": v for k, v in part.items()})
    store["metadata"] = {
        "dataset": cfst.get_spec(args.dataset).name,
        "n_slots": args.n_slots,
        "phi_dim": int(store["continual_train_phi"].shape[-1]),
        "n_classes": cfst.get_spec(args.dataset).num_classes,
        "ckpt": CHECKPOINT_URL if dino == "b14" else str(args.checkpoint),
        "seed": args.seed,
        "patch_source": source.__dict__ if source else "dinov2_b14",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{source.name}" if source else ""
    path = args.out / f"{args.dataset}_slots_K{args.n_slots}{tag}.pt"
    torch.save(store, path)
    print("saved", path)


if __name__ == "__main__":
    main()
