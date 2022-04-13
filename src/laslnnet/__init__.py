import torch
import torch.nn as nn
import numpy as np

from torch import Tensor

from . import pyst
from . import xyston as xy


class Dost2:
    def __init__(self, im):
        self._N = len(im)
        self._n = 2 * int(np.log2(self._N)) - 1
        self._b = pyst.dst2(im)

    def __array__(self, dtype=complex):
        a = np.zeros((self._N, self._N, self._n, self._n), dtype=complex)
        for x in range(self._N):
            for y in range(self._N):
                a[x, y, :, :] = pyst.freqdomain(self._b, x, y)
        if dtype == complex:
            return a
        a = np.array([a])
        return np.stack((a.real, a.imag)).astype(dtype)

    def __iter__(self):
        for x in range(self._N):
            for y in range(self._N):
                yield pyst.freqdomain(self._b, x, y)

    def __len__(self):
        return self._N


class LASLNNet(nn.Module):
    def __init__(self) -> None:
        super(LASLNNet, self).__init__()
        self.conv = nn.Sequential(
            xy.CConv4d(1, 32, kernel_size=3, stride=2),
            xy.CReLU(),
            xy.CConv4d(32, 64, kernel_size=3, padding="same"),
            xy.CReLU(),
            xy.CConv4d(64, 128, kernel_size=1, padding="same"),
            xy.CReLU(),
            xy.CConv4d(128, 128, kernel_size=1, padding="same"),
            xy.CReLU(),
            xy.CConv4d(128, 64, kernel_size=1, stride=2),
            xy.CReLU(),
            nn.Flatten(start_dim=2),
            xy.CReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        fc = nn.Linear(in_features=x.shape[2], out_features=1)
        return fc(x)


def train_cnn():
    pass
