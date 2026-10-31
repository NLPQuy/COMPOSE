"""Few-shot class-incremental benchmarks (Table 10), with the CEC session splits.

* ``cifar100``: 60 base classes + 8 sessions of 5-way 5-shot; torchvision CIFAR-100.
* ``cub200``: 100 base classes + 10 sessions of 10-way 5-shot; ``<root>/CUB_200_2011``.
* ``mini_imagenet``: 60 base classes + 8 sessions of 5-way 5-shot; Hugging Face
  ``timm/mini-imagenet`` (train / test splits).

The few-shot images of each incremental session come from the CEC index lists
(https://github.com/icoz69/CEC-CVPR2021, ``data/index_list/<dataset>/session_<s>.txt``).
The Hugging Face copy of miniImageNet does not keep file names, so its sessions use the
first five training images of each new class (same classes as CEC, whose class order
equals the sorted synset order of the Hugging Face labels). Base training uses every
training image of the base classes; each test set holds all classes seen so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

TRANSFORM = transforms.Compose([      # the DINOSAUR input pipeline
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@dataclass(frozen=True)
class FSCILSpec:
    base_classes: int
    num_classes: int
    way: int
    shot: int

    @property
    def sessions(self) -> int:
        return 1 + (self.num_classes - self.base_classes) // self.way


SPECS = {
    "cifar100": FSCILSpec(60, 100, 5, 5),
    "cub200": FSCILSpec(100, 200, 10, 5),
    "mini_imagenet": FSCILSpec(60, 100, 5, 5),
}


class _HFImage:
    """Picklable accessor for one image of a Hugging Face split (DataLoader workers)."""

    def __init__(self, split):
        self.split = split

    def __call__(self, i):
        return self.split[i]["image"]


class ImageList(Dataset):
    """``items`` are paths, uint8 arrays, or keys passed to ``fetch`` (loaded lazily)."""

    def __init__(self, items: list, labels: list[int], fetch=None):
        self.items, self.labels, self.fetch = items, labels, fetch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        if self.fetch is not None:
            img = self.fetch(x)
        elif isinstance(x, np.ndarray):
            img = Image.fromarray(x)
        else:
            img = Image.open(x)
        return TRANSFORM(img.convert("RGB")), self.labels[i]


class FSCILData:
    """Image lists for the base split, each incremental session, and the cumulative tests."""

    def __init__(self, name: str, root: str | Path | None, index_dir: str | Path):
        self.name, self.spec = name, SPECS[name]
        self.root = Path(root).expanduser() if root else None
        self.index_dir = Path(index_dir) / name
        getattr(self, f"_load_{name}")()

    def _session_file(self, s: int) -> list[str]:
        return (self.index_dir / f"session_{s + 1}.txt").read_text().splitlines()

    # -- per dataset ------------------------------------------------------------------
    def _load_cifar100(self):
        self._train = datasets.CIFAR100(str(self.root), train=True, download=True)
        self._test = datasets.CIFAR100(str(self.root), train=False, download=True)

    def _load_cub200(self):
        base = str(self.root)
        read = lambda f: dict(line.split(" ", 1) for line in Path(f"{base}/CUB_200_2011/{f}").read_text().splitlines())
        images, split, classes = read("images.txt"), read("train_test_split.txt"), read("image_class_labels.txt")
        self._cub = {True: ([], []), False: ([], [])}
        self._cub_label = {}
        for k in sorted(images, key=int):
            path = f"{base}/CUB_200_2011/images/{images[k]}"
            label = int(classes[k]) - 1
            self._cub[split[k] == "1"][0].append(path)
            self._cub[split[k] == "1"][1].append(label)
            self._cub_label[path] = label

    def _load_mini_imagenet(self):
        from datasets import load_dataset

        self._hf = {s: load_dataset("timm/mini-imagenet", split=s) for s in ("train", "test")}
        self._syn2lbl = {n: i for i, n in enumerate(self._hf["train"].features["label"].names)}

    # -- splits -----------------------------------------------------------------------
    def _by_classes(self, classes: set[int], train: bool) -> ImageList:
        if self.name == "cifar100":
            ds = self._train if train else self._test
            idx = [i for i, t in enumerate(ds.targets) if t in classes]
            return ImageList([ds.data[i] for i in idx], [ds.targets[i] for i in idx])
        if self.name == "cub200":
            paths, labels = self._cub[train]
            labels = np.asarray(labels)
            keep = [j for c in sorted(classes) for j in np.where(labels == c)[0]]
            return ImageList([paths[j] for j in keep], [int(labels[j]) for j in keep])
        split = "train" if train else "test"
        labels = self._hf[split]["label"]
        idx = [i for i, y in enumerate(labels) if y in classes]
        return ImageList(idx, [labels[i] for i in idx], fetch=_HFImage(self._hf[split]))

    def base_train(self) -> ImageList:
        return self._by_classes(set(range(self.spec.base_classes)), train=True)

    def session_train(self, s: int) -> ImageList:
        """The few-shot images of incremental session ``s >= 1``."""
        lines = self._session_file(s)
        if self.name == "cifar100":
            idx = [int(x) for x in lines]
            return ImageList([self._train.data[i] for i in idx], [self._train.targets[i] for i in idx])
        if self.name == "cub200":
            paths = [f"{self.root}/{line}" for line in lines]
            return ImageList(paths, [self._cub_label[p] for p in paths])
        wanted = {self._syn2lbl[line.split("/")[2]] for line in lines}
        hf, labels, counts, idx = self._hf["train"], self._hf["train"]["label"], {}, []
        for i, y in enumerate(labels):
            if y in wanted and counts.get(y, 0) < self.spec.shot:
                counts[y] = counts.get(y, 0) + 1
                idx.append(i)
        return ImageList(idx, [labels[i] for i in idx], fetch=_HFImage(hf))

    def test_until(self, s: int) -> ImageList:
        """Test images of every class seen up to session ``s`` (0 = base)."""
        return self._by_classes(set(range(self.spec.base_classes + s * self.spec.way)), train=False)
