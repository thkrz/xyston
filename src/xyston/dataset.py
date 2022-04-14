import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from . import signal


class LASLDataset(Dataset):
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
        image = signal.real(signal.dost2(image))
        label = self.img_labels[i][0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
