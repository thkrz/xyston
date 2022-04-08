import torch
import torch.nn as nn


class LASLNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(),
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
