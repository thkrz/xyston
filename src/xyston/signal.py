import numpy as np

from . import fstpack


def conformal_hilbert(im, kernel_size=7, padding="valid"):
    if not isinstance(im, np.ndarray):
        im = im.__array__()
    if padding == "valid":
        work = np.pad(im, kernel_size)
    h = np.zeros((4, im.shape[0], im.shape[1]))
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            h[:, x, y] = fstpack.cmsht2(
                work, x + kernel_size, y + kernel_size, kernel_size=kernel_size
            )
    return h


def dost2(im, dtype=np.csingle):
    if not isinstance(im, np.ndarray):
        im = im.__array__()
    N = len(im)
    assert N != 0 and (N & (N - 1)) == 0
    n = 2 * int(np.log2(N)) - 1
    b = fstpack.dst2(im)
    arr = np.zeros((N, N, n, n), dtype=dtype)
    for x in range(N):
        for y in range(N):
            arr[x, y, :, :] = fstpack.imfreq(b, x, y)
    return arr


def cmplx(arr):
    if not isinstance(arr, np.ndarray):
        arr = arr.__array__()
    if arr.dtype == complex:
        return arr
    # TODO: stack real and imag


def real(arr, dtype=np.float32):
    if not isinstance(arr, np.ndarray):
        arr = arr.__array__()
    if arr.dtype != complex:
        return arr
    arr = np.array([arr])
    return np.stack((arr.real, arr.imag)).astype(dtype)
