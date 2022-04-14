import numpy as np

from . import pyst


def dost2(im):
    if not isinstance(im, np.ndarray):
        im = im.__array__()
    N = len(im)
    assert N != 0 and (N & (N - 1)) == 0
    n = 2 * int(np.log2(N)) - 1
    b = pyst.dst2(im)
    arr = np.zeros((N, N, n, n), dtype=complex)
    for x in range(N):
        for y in range(N):
            arr[x, y, :, :] = pyst.freqdomain(b, x, y)
    return arr


def cmplx(arr):
    if not isinstance(arr, np.ndarray):
        arr = arr.__array__()
    if arr.dtype == complex:
        return arr
    # TODO: stack real and imag


def real(arr):
    if not isinstance(arr, np.ndarray):
        arr = arr.__array__()
    if arr.dtype != complex:
        return arr
    arr = np.array([arr])
    return np.stack((arr.real, arr.imag)).astype(float)
