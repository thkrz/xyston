import numpy as np
import xyston


def chirp22():
    n = 64
    m = n // 2
    h = np.zeros((n, n))
    for x in range(n):
        f = 10.0 * np.cos(2.0 * np.pi * (0.15 * x) * x / 64.0)
        for y in range(n):
            xx = (x - m) ** 2
            yy = (y - m) ** 2
            h[x, y] = f * np.exp(-0.5 * (xx / n + yy / m))
    return h


if __name__ == "__main__":
    h = chirp22()
    dost = xyston.DOST(h)

    for im in dost:
        print(im)