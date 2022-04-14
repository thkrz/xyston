import numpy as np
import random
from pathlib import Path

_data_root = Path("/home/thk32is/work/laslds")
_tr = {"__LASL__": 1, "__NONE__": 0}


def load(batch_size=4, split="train"):
    path = _data_root / split
    df = {}
    with open(path / "map.txt") as f:
        for ln in f:
            s = ln.split()
            k = s[1].strip()
            v = s[0]
            df[k] = v
    keys = list(df.keys())
    random.shuffle(keys)
    labels, images = [], []
    for k in keys:
        labels.append(_tr[df[k]])
        arr = np.loadtxt(path / (k + ".gz"))
        images.append(arr)
    return np.array(labels, dtype=int), np.array(images)
