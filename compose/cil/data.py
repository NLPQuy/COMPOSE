"""Standard class-incremental benchmarks (Table 11), following LAMDA-PILOT's ``utils/data*.py``.

* ``cifar224``: CIFAR-100 at 224x224, 10 base classes + 9 sessions of 10.
* ``imagenetr`` / ``imageneta``: ImageNet-R / ImageNet-A as ``<root>/{train,test}/<class>/``,
  20 base classes + 9 sessions of 20.

Images are resized and centre-cropped to 224 and kept in [0, 1] (no mean/std
normalisation), as in LAMDA-PILOT. Classes keep their dataset order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CILSpec:
    num_classes: int
    init_cls: int
    increment: int

    @property
    def sessions(self) -> list[int]:
        """Number of new classes in each session."""
        sizes = [self.init_cls]
        while sum(sizes) + self.increment < self.num_classes:
            sizes.append(self.increment)
        if self.num_classes > sum(sizes):
            sizes.append(self.num_classes - sum(sizes))
        return sizes


SPECS = {
    "cifar224": CILSpec(100, 10, 10),
    "imagenetr": CILSpec(200, 20, 20),
    "imageneta": CILSpec(200, 20, 20),
}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


class ImageSet(Dataset):
    """Images (arrays or paths) with labels; yields ``(index, image, label)`` like LAMDA-PILOT."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        img = Image.open(img).convert("RGB") if isinstance(img, (str, np.str_)) else Image.fromarray(img)
        return i, self.transform(img), self.labels[i]


class CILData:
    """Train/test images of one benchmark; ``subset`` selects a range of classes."""

    def __init__(self, name: str, root: str | Path):
        self.name, self.spec = name, SPECS[name]
        if name == "cifar224":
            train = datasets.CIFAR100(str(root), train=True, download=True)
            test = datasets.CIFAR100(str(root), train=False, download=True)
            self.train = (train.data, np.array(train.targets))
            self.test = (test.data, np.array(test.targets))
        else:
            self.train = self._folder(Path(root) / "train")
            self.test = self._folder(Path(root) / "test")

    @staticmethod
    def _folder(path: Path) -> tuple[np.ndarray, np.ndarray]:
        imgs = datasets.ImageFolder(str(path)).imgs
        return np.array([p for p, _ in imgs]), np.array([y for _, y in imgs])

    def subset(self, classes: range, source: str, train_transform: bool = False) -> ImageSet:
        """Images of ``classes``, grouped class by class (LAMDA-PILOT ``get_dataset`` order)."""
        x, y = self.train if source == "train" else self.test
        idx = np.concatenate([np.where(y == c)[0] for c in classes])
        return ImageSet(x[idx], y[idx], TRAIN_TRANSFORM if train_transform else TEST_TRANSFORM)

    def session_ranges(self) -> list[range]:
        out, start = [], 0
        for size in self.spec.sessions:
            out.append(range(start, start + size))
            start += size
        return out
