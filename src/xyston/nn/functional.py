import torch
from torch import Tensor


def modulus(input: Tensor, dim=1) -> Tensor:
    r = torch.index_select(input, dim, 0)
    i = torch.index_select(input, dim, 1)
    return torch.sqrt(r**2 + i**2)
