from pathlib import Path

data_root = Path("./LASLNNET")


def loadtrain(batch_size=4):
    path = data_root / "train"
    df = {"__LASL__": [], "__NONE__": []}
    with open(path / "map.txt") as f:
        for ln in f:
            s = ln.split()
            df[s[0]] = s[1].strip()
