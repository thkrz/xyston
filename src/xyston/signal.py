import numpy as np
from collections import namedtuple

from . import fstpack

monogenic_order = ["curvature", "direction", "phase", "energy"]
MonogenicSignal = namedtuple("MonogenicSignal", monogenic_order)


class MonogenicImage:
    def __init__(
        self,
        image,
        kernel_size: int = 7,
        padding: str = "valid",
        coarse: float = 0.2,
        fine: float = 0.1,
    ):
        if not isinstance(image, np.ndarray):
            image = image.__array__()
        self.image = image
        if isinstance(padding, str):
            assert padding in ["valid"]
        self.kernel_size = kernel_size
        self.padding = padding
        self.coars = 0.2
        self.fine = 0.1

        if isinstance(padding, str):
            pad_ = 0 if padding == "valid" else None
        else:
            pad_ = padding
        if pad_ is None:
            self.work = image
        else:
            self.work = np.pad(image, kernel_size, constant_values=pad_)

    def __array__(self, dtype=np.float32):
        shape = self.image.shape
        arr = np.zeros((4, shape[0], shape[1]), dtype=dtype)
        for x in range(shape[0]):
            for y in range(shape[1]):
                arr[:, x, y] = fstpack.cmsht2(
                    self.work,
                    x + self.kernel_size,
                    y + self.kernel_size,
                    kernel_size=self.kernel_size,
                    coarse=self.coarse,
                    fine=self.fine,
                )
        return arr

    def __getitem__(self, xy):
        x, y = xy
        h = fstpack.cmsht2(
            self.work,
            x + self.kernel_size,
            y + self.kernel_size,
            kernel_size=self.kernel_size,
            coarse=self.coarse,
            fine=self.fine,
        )
        return MonogenicSignal(*h)

    def __iter__(self):
        shape = self.image.shape
        for x in range(shape[0]):
            for y in range(shape[1]):
                yield self.__getitem__((x, y))

    def __len__(self):
        return len(self.image)


class DostImage:
    def __init__(self, image):
        if not isinstance(image, np.ndarray):
            image = image.__array__()
        N = len(image)
        assert N != 0 and (N & (N - 1)) == 0
        self._N = N
        self._n = 2 * int(np.log2(N)) - 1
        self.base = fstpack.dst2(image)

    def __array__(self, dtype=np.csingle):
        arr = np.zeros((self._N, self._N, self._n, self._n), dtype=dtype)
        for x in range(self._N):
            for y in range(self._N):
                arr[x, y, :, :] = fstpack.imfreq(self.base, x, y)
        return arr

    def __getitem__(self, xy):
        return fstpack.imfreq(self.base, *xy)

    def __iter__(self):
        for x in range(self._N):
            for y in range(self._N):
                yield fstpack.imfreq(self.base, x, y)

    def __len__(self):
        return self._N


def cmplx(arr, dtype=np.csingle):
    if not isinstance(arr, np.ndarray):
        arr = arr.__array__()
    if arr.dtype == dtype:
        return arr
    # TODO: stack real and imag


def real(arr, dtype=np.float32):
    if not isinstance(arr, np.ndarray):
        arr = arr.__array__()
    if arr.dtype == dtype:
        return arr
    arr = np.array([arr])
    return np.stack((arr.real, arr.imag)).astype(dtype)
