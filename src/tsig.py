import numpy as np
import xyston as xy

from pathlib import Path

train_dir = Path("../train")

if __name__ == "__main__":
    for f in (train_dir / "p").iterdir():
        im = np.loadtxt(f)
        S = xy.dost(im)
        break
    print(S[1, ::8, 1, 1])
