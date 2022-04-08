import numpy as np

from . import pyst


class dost:
    def __init__(self, im):
        self.N = len(im)
        self.n = 2 * int(np.log2(self.N)) - 1
        self.shape = (self.N, self.N, self.n, self.n)
        self.size = self.N**2 * self.n**2
        self.base = pyst.dst2(im)

    def __getitem__(self, pos):
        x, y, vx, vy = pos
        S = pyst.freqdomain(self.base, x, y)
        return S[vx, vy]

    def srange(self, s):
        start = s.start or 0
        stop = s.stop or self.N
        step = s.step or 1
        return np.arange(start, stop, step)
