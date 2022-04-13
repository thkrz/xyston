import numpy as np

from . import pyst


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


def train_cnn():
    a = np.zeros((8, 8))
    S = Dost2(a)
    b = np.array(S, dtype=float)
    print(b.shape)
