import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_4_t
from typing import Optional


def conv4d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: _size_4_t = 1,
    padding: _size_4_t = 0,
    dilation: _size_4_t = 1,
    groups=1,
) -> Tensor:
    l_i = input.shape[2]
    l_k = weight.shape[2]
    if isinstance(padding, tuple):
        pad_ = padding[0]
        padding_ = padding[1:]
    else:
        padding_ = padding
        pad_ = 0 if padding == "valid" else None

    if pad_ is None:
        l_o = l_i
    else:
        l_o = (l_i + 2 * pad_ - dilation[0] * (l_k - 1) - 1) // stride[0] + 1
    o_f = l_o * [None]
    for i in range(l_k):
        for j in range(l_i):
            n = j - (i - l_k // 2) - (l_i - l_o) // 2
            if n < 0 or n >= l_o:
                continue
            f = F.conv3d(
                input[:, :, j, :],
                weight[:, :, i, :, :],
                bias,
                stride[1:],
                padding_,
                dilation[1:],
                groups,
            )
            if o_f[n] is None:
                o_f[n] = f
            else:
                o_f[n] += f
    return torch.stack(o_f, dim=2)


def modulus(input: Tensor) -> Tensor:
    r = input[:, 0]
    i = input[:, 1]
    return torch.sqrt(r**2 + i**2)
