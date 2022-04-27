import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .. import functional as G


class CReLU(nn.Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(CReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        r = F.relu(input[:, 0], inplace=self.inplace)
        i = F.relu(input[:, 1], inplace=self.inplace)
        if self.inplace:
            return input
        return torch.stack((r, i), dim=1)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class zReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return G.zrelu(input)


class Modulus(nn.Module):
    def __init__(self) -> None:
        super(Modulus, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return G.modulus(input)
