import torch.nn as nn
from torch import Tensor

from .. import nn as xy


class LASLNet34(nn.Module):
    def __init__(self) -> None:
        super(LASLNet34, self).__init__()
        self.conv = nn.Sequential(
            xy.CConv3d(1024, 2048, kernel_size=3, stride=2),
            xy.CReLU(),
            xy.CConv3d(2048, 4096, kernel_size=3, padding="same"),
            xy.CReLU(),
            xy.CConv3d(4096, 4096, kernel_size=1, padding="same"),
            xy.CReLU(),
            xy.CConv3d(4096, 2048, kernel_size=1, padding="same"),
            xy.CReLU(),
            nn.Flatten(start_dim=2),
            xy.CReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv(input)
        fc = nn.Linear(in_features=input.shape[2], out_features=1)
        return fc(input)


class LASLNet45(nn.Module):
    def __init__(self) -> None:
        super(LASLNet45, self).__init__()
        self.conv = nn.Sequential(
            xy.CConv4d(1, 32, kernel_size=3, stride=2),
            xy.CReLU(),
            xy.CConv4d(32, 64, kernel_size=3, padding="same"),
            xy.CReLU(),
            xy.CConv4d(64, 128, kernel_size=1, padding="same"),
            xy.CReLU(),
            xy.CConv4d(128, 128, kernel_size=1, padding="same"),
            xy.CReLU(),
            xy.CConv4d(128, 64, kernel_size=1, stride=2),
            xy.CReLU(),
            nn.Flatten(start_dim=2),
            xy.CReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv(input)
        fc = nn.Linear(in_features=input.shape[2], out_features=1)
        return fc(input)
