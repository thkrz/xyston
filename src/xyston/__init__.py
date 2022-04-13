import numpy as np

from . import pyst


class Dost2:
    def __init__(self, im):
        self._N = len(im)
        self._n = 2 * int(np.log2(self._N)) - 1
        self._b = pyst.dst2(im)

    def __array__(self, dtype=complex):
        arr = np.zeros((self._N, self._N, self._n, self._n), dtype=dtype)
        for x in range(self._N):
            for y in range(self._N):
                arr[x, y, :, :] = pyst.freqdomain(self._b, x, y)
        return arr

    def __iter__(self):
        for x in range(self._N):
            for y in range(self._N):
                yield pyst.freqdomain(self._b, x, y)

    def __len__(self):
        return self._N


def cmplx(arr):
    pass


def real(arr):
    if not isinstance(arr, np.ndarray):
        arr = arr.__array__()
    arr = np.array([arr])
    return np.stack((arr.real, arr.imag)).astype(float)


def xyston_main():
    a = np.zeros((8, 8))
    S = Dost2(a)
    c = real(S)
    print(c.shape)
    return 0
