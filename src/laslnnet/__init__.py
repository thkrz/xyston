import torch
import torch.nn as nn
import numpy as np

from torch import Tensor

from .chirp import chirp
from . import pyst
from . import xyston as xy


class Dost2:
    def __init__(self, im):
        self._N = len(im)
        self._n = 2 * int(np.log2(self._N)) - 1
        self._b = pyst.dst2(im)

    def __array__(self, dtype=complex):
        a = np.zeros((self._N, self._N, self._n, self._n), dtype=dtype)
        for x in range(self._N):
            for y in range(self._N):
                a[x, y, :, :] = pyst.freqdomain(self._b, x, y)
        return a

    def __iter__(self):
        for x in range(self._N):
            for y in range(self._N):
                yield pyst.freqdomain(self._b, x, y)

    def totensor(self):
        X = np.array([[self.__array__()]])
        X = np.stack((X.real, X.imag), axis=1)
        return torch.Tensor(X)


class LASLNNet(nn.Module):
    def __init__(self, dim: int) -> None:
        super(LASLNNet, self).__init__()
        self.conv = nn.Sequential(
            xy.CConv4d(1, 2, kernel_size=2),
            xy.CReLU(inplace=True),
            xy.CConv4d(2, 4, kernel_size=2),
            xy.CReLU(inplace=True),
            xy.CConv4d(4, 8, kernel_size=2),
            xy.CReLU(inplace=True),
            xy.CConv4d(8, 16, kernel_size=2),
            xy.CReLU(inplace=True),
            xy.CConv4d(16, 32, kernel_size=2),
            xy.CReLU(inplace=True),
            xy.CConv4d(32, 64, kernel_size=2),
            xy.CReLU(inplace=True),
        )
        n = 2 * int(np.log2(dim)) - 7
        N = 64 * (dim - 6) ** 2 * n**2
        self.fc = nn.Linear(in_features=N, out_features=8)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.flatten(start_dim=2, end_dim=-1)
        x = self.fc(x)
        return x


def test_cnn():
    h = chirp()
    S = Dost2(h)
    X = S.totensor()
    Y = xy.CReLU(inplace=True).forward(X)
    lnet = LASLNNet(h.shape[0])
    Y = lnet.forward(X)
    print(Y)
