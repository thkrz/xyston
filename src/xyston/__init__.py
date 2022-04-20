import numpy as np
from sklearn.metrics import classification_report
import torch

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import torch.nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from . import dataset
from .nn.model import LASLNet45


def preprocess(x, y):
    return x.to(dev), y.float().to(dev)


def train(model, data, path):
    optim = Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.BCELoss()
    scheduler = ExponentialLR(optim, gamma=0.9)
    train_dl = DataLoader(data, batch_size=30, shuffle=True)
    train_dl = dataset.WrappedDataLoader(train_dl, preprocess)
    for epoch in range(20):
        for input, target in train_dl:
            optim.zero_grad()
            loss = loss_fn(model(input), target)
            loss.backward()
            optim.step()
        scheduler.step()
    torch.save(model.state_dict(), path)


def val(model, data, path=None):
    val_dl = DataLoader(data, batch_size=30, shuffle=True)
    val_dl = dataset.WrappedDataLoader(val_dl, preprocess)
    y_true = []
    y_pred = []
    with torch.no_grad():
        if path:
            model.load_state_dict(torch.load(path))
            model.eval()
        for input, target in val_dl:
            pred = model(input)
            y_true += target.tolist()
            y_pred += pred.tolist()
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    print(classification_report(y_true, y_pred))


def xyston_main():
    model_path = "../laslnet45.pt"
    model = LASLNet45()
    model.to(dev)
    print(f"Model on {dev}")
    lasl_ds = dataset.LASLDataset("../../../laslds")
    print(f"Start training with {len(lasl_ds.train_data)} records")
    train(model, lasl_ds.train_data, model_path)
    val(model, lasl_ds.val_data)
    return 0
