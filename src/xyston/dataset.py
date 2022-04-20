import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from . import signal


def _target_transform(s):
    tr = {"__LASL__": 1.0, "__NONE__": 0.0}
    return np.float32(tr[s])


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class LASLDataset:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self._load("train")
        self._load("val")
        self._load("test")

    def _load(self, name):
        cwd = self.base_dir / name
        self.__dict__[name + "_data"] = STDataset(
            cwd / "map.txt", cwd, target_transform=_target_transform
        )


class STDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        with open(annotations_file) as f:
            self.img_labels = [s.split() for s in f]
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        img_path = self.img_dir / (self.img_labels[i][1] + ".gz")
        image = np.loadtxt(img_path)
        image = signal.real(signal.dost2(image)).astype(np.float32)
        label = self.img_labels[i][0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
