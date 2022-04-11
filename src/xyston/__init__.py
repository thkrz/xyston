import numpy as np

from . import pyst


class dost2:
    def __bnit__(self, im):
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
