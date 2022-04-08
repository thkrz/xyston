import numpy as np
import xyston

from pathlib import Path

train_dir = Path("../train")

if __name__ == "__main__":
    pos = []
    for f in (train_dir / "p").iterdir():
        im = np.loadtxt(f)
        dost = xyston.DOST(im)
        pos.append(list(dost))

    pos = np.array(pos)
    print(pos.shape)

