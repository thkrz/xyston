import torch
from torch import Tensor


def modulus(input: Tensor, dim: int = 1) -> Tensor:
    r = input[:, 0]
    i = input[:, 1]
    return torch.sqrt(r**2 + i**2)
