import numpy as np


def chirp1d_1():
    h = np.zeros(128)
    for t in range(64):
        h[t] = np.cos(2.0 * np.pi * t * 6.0 / 128.0)
    for t in range(64, 128):
        h[t] = np.cos(2.0 * np.pi * t * 25.0 / 128.0)
    for t in range(20, 30):
        h[t] += 0.5 * np.cos(2.0 * np.pi * t * 52.0 / 128.0)
    return h


def chirp1d_2():
    h = np.zeros(256)
    for t in range(256):
        h[t] = np.cos(2.0 * np.pi * (10.0 + t / 7.0) * t / 256.0) + np.cos(
            2.0 * np.pi * (256.0 / 2.8 - t / 6.0) * t / 256.0
        )
    for t in range(114, 122):
        h[t] += np.cos(2.0 * np.pi * t * 0.42)
    for t in range(134, 142):
        h[t] += np.cos(2.0 * np.pi * t * 0.42)
    return h


def chirp2d():
    h = np.zeros((64, 16))
    for x in range(64):
        h[x, :] = 10.0 * np.cos(2.0 * np.pi * 4.0 * x / 64.0)
    for x in range(33):
        h[x, :] += 10.0 * np.cos(2.0 * np.pi * 24.0 * x / 64.0)
    for x in range(33, 64):
        h[x, :] += 10.0 * np.cos(2.0 * np.pi * 10.0 * x / 64.0)
    return h


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2, "missing signal identifier"
    h = {"1d-1": chirp1d_1, "1d-2": chirp1d_2, "2d": chirp2d}[sys.argv[1]]()
    np.savetxt(sys.stdout.buffer, h, fmt="%11.6f")
