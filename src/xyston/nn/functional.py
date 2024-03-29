import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t, _size_4_t
from torch.nn.modules.utils import _quadruple
from typing import Optional

from .utils import output_shape, pooling_output_shape, zeros_like


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
    shape = pooling_output_shape(input.shape, padding, kernel_size, stride, ceil_mode)
    o_t = zeros_like(input, shape=shape)
    for i in range(shape[2]):
        for j in range(kernel_size[0]):
            n = stride[0] * i + j
            if n >= input.shape[2]:
                continue
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
    l_o = output_shape(l_i, pad_, dilation[0], l_k, stride[0])
    o_t = l_o * [None]
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
            if o_t[n] is None:
                o_t[n] = f
            else:
                o_t[n] += f
    return torch.stack(o_t, dim=2)


def max_pool4d(
    input: Tensor,
    kernel_size: _size_4_t,
    stride: _size_4_t = None,
    padding: _size_4_t = 0,
    dilation: _size_4_t = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    kernel_size = _quadruple(kernel_size)
    stride = kernel_size if stride is None else _quadruple(stride)
    padding = _quadruple(padding)
    dilation = _quadruple(dilation)
    shape = output_shape(input.shape, padding, dilation, kernel_size, stride, ceil_mode)
    o_t = zeros_like(input, shape=shape)
    if return_indices:
        i_t = zeros_like(input, shape=shape, dtype=int)
    k_t = zeros_like(input, shape=shape)
    for i in range(shape[2]):
        for j in range(kernel_size[0]):
            n = stride[0] * i + j
            if n >= input.shape[2]:
                continue
            k_t[:, :, j] = F.max_pool3d(
                input[:, :, n],
                kernel_size[1:],
                stride[1:],
                padding[1:],
                dilation[1:],
                ceil_mode,
                False,
            )
        m = torch.max(k_t, 2, keepdim=True)
        o_t[:, :, i] = m.values[:, :, 0]
        if return_indices:
            i_t[:, :, i] = m.indices[:, :, 0]
    del k_t
    if return_indices:
        return o_t, i_t
    return o_t


def window_select2d(
    input: Tensor,
    kernel_size: _size_2_t,
    stride: _size_2_t,
    out: Tensor,
) -> Tensor:
    m = kernel_size[0] // 2
    n = kernel_size[1] // 2
    for h in range(out.shape[2]):
        for w in range(out.shape[3]):
            out[:, :, h, w] = input[:, :, stride[0] * h + m, stride[1] * w + n]
    return out


def modulus(input: Tensor) -> Tensor:
    r = input[:, 0]
    i = input[:, 1]
    return torch.sqrt(r**2 + i**2)


def zrelu(input: Tensor) -> Tensor:
    r = input[:, 0]
    i = input[:, 1]
    arg = ((r > 0) & (i > 0)).int()
    return torch.stack((r * arg, i * arg), dim=1)
