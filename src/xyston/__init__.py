import numpy as np
import torch

from . import dataset
from . import signal
from .nn import model


def batch_size(n, min_size=2, max_size=32):
    for i in range(max_size, min_size - 1, -1):
        if n % i == 0:
            return i
    return 0


def train(ims, lbl):
    m = model.LASLNet45()
    y = m(ims)
    print(y.shape)


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
            a = torch.tensor(np.array(batch))
            l = lbl[n]
            train(a, l)
            n += 1
            batch = []
            break
    return 0
