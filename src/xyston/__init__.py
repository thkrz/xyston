import numpy as np
from torch import tensor

from . import dataset
from . import signal
from .nn import model


def batch_size(n, min_size=2, max_size=32):
    assert min_size > 1
    for i in range(max_size, min_size - 1, -1):
        if n % i == 0:
            return i
    assert i >= min_size


def train(ims, lbl):
    m = model.LASLNet45()
    m.cuda()
    return m(tensor(ims).float())


def xyston_main():
    lbl, ims = dataset.load()
    N = ims.shape[0]
    b = batch_size(N)
    lbl = lbl.reshape(-1, b)
    batch = []
    n = 0
    for im in ims:
        batch.append(signal.real(signal.dost2(im)))
        if len(batch) == b:
            a = np.array(batch)
            l = lbl[n]
            train(a, l)
            n += 1
            batch = []
    return 0
