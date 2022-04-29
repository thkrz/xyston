import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_4_t
from typing import Optional


def _output_size(
    size: int, padding: int, dilation: int, kernel_size: int, stride: int
) -> int:
    if padding is None:
        return size
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def _pooling_size(
    size: int, padding: int, kernel_size: int, stride: int, ceil_mode: bool
) -> int:
    num = size + 2 * padding - kernel_size
    if ceil_mode:
        return -(-num // stride) + 1
    return num // stride + 1


def avg_pool4d(
    input: Tensor,
    kernel_size: _size_4_t,
    stride: _size_4_t = None,
    padding: _size_4_t = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
) -> Tensor:
    if stride is None:
        stride = kernel_size
    divisor = kernel_size[0] if divisor_override is None else divisor_override
    b_i, c_i, l_i, d_i, h_i, w_i = input.shape
    l_o = _pooling_size(l_i, padding[0], kernel_size[0], stride[0], ceil_mode)
    d_o = _pooling_size(d_i, padding[1], kernel_size[1], stride[1], ceil_mode)
    h_o = _pooling_size(h_i, padding[2], kernel_size[2], stride[2], ceil_mode)
    w_o = _pooling_size(w_i, padding[3], kernel_size[3], stride[3], ceil_mode)
    o_t = torch.zeros(
        (b_i, c_i, l_o, d_o, h_o, w_o),
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )
    for i in range(l_o):
        for j in range(kernel_size[0]):
            n = stride[0] * i + j
            o_t[:, :, i] += F.avg_pool3d(
                input[:, :, n],
                kernel_size[1:],
                stride[1:],
                padding[1:],
                ceil_mode,
                count_include_pad,
                divisor_override,
            )
        o_t[:, :, i] /= divisor
    return o_t


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
    l_o = _output_size(l_i, pad_, dilation[0], l_k, stride[0])
    o_f = l_o * [None]
    for i in range(l_k):
        for j in range(l_i):
            n = j - (i - l_k // 2) - (l_i - l_o) // 2
            if n < 0 or n >= l_o:
                continue
            f = F.conv3d(
                input[:, :, j],
                weight[:, :, i],
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


def zrelu(input: Tensor) -> Tensor:
    r = input[:, 0]
    i = input[:, 1]
    arg = ((r > 0) & (i > 0)).int()
    return torch.stack((r * arg, i * arg), dim=1)
