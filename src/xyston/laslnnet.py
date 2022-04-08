import torch
import torch.nn as nn


class LASLNNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(n, n//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear()

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return {"out": x}
