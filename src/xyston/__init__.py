from . import pyst


class DOST:
    def __init__(self, im):
        self.N = len(im)
        self.S = pyst.dst2(im)

    def __getitem__(self, pos):
        x, y = pos
        assert 0 <= x < self.N and 0 <= y < self.N
        return pyst.freqdomain(self.S, x, y)

    def __iter__(self):
        for x in range(self.N):
            for y in range(self.N):
                yield self.__getitem__((x, y))
