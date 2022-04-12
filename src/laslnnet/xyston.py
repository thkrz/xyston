import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.common_types import _size_4_t
from torch.nn.modules.utils import _quadruple

from typing import Optional, Union


class Conv4d(nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_4_t,
        stride: _size_4_t = 1,
        padding: Union[str, _size_4_t] = 0,
        dilation: _size_4_t = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _quadruple(kernel_size)
        stride_ = _quadruple(stride)
        padding_ = padding if isinstance(padding, str) else _quadruple(padding)
        dilation_ = _quadruple(dilation)
        super(Conv4d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _quadruple(0),
            groups,
            bias,
            "zeros",
            **factory_kwargs,
        )

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        b, c_i, l_i, d_i, h_i, w_i = input.shape
        l_k = self.kernel_size[0]
        if isinstance(self.padding, tuple):
            pad_ = self.padding[0]
            padding_ = self.padding[1:]
        else:
            padding_ = self.padding
            pad_ = 0 if self.padding == "valid" else -1
        l_o = (
            l_i
            if pad_ < 0
            else (l_i + 2 * pad_ - self.dilation[0] * (l_k - 1) - 1) // self.stride[0]
            + 1
        )
        o_f = l_o * [None]
        for i in range(l_k):
            for j in range(l_i):
                n = j - (i - l_k // 2) - (l_i - l_o) // 2
                if n < 0 or n >= l_o:
                    continue
                f = F.conv3d(
                    input[:, :, j, :].view(b, c_i, d_i, h_i, w_i),
                    weight[:, :, i, :, :],
                    bias,
                    self.stride[1:],
                    padding_,
                    self.dilation[1:],
                    self.groups,
                )
                if o_f[n] is None:
                    o_f[n] = f
                else:
                    o_f[n] += f
        return torch.stack(o_f, dim=2)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class _CConvNd(nn.Module):
    def __init__(self, cls, *args, **kwargs) -> None:
        super(_CConvNd, self).__init__()
        self.conv = cls(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        m_r = self.conv(input[:, 0])
        m_i = self.conv(input[:, 1])
        r = m_r - m_i
        i = m_i + m_r
        return torch.stack((r, i), dim=1)


def CConv3d(*args, **kwargs):
    return _CConvNd(nn.Conv3d, *args, **kwargs)


def CConv4d(*args, **kwargs):
    return _CConvNd(Conv4d, *args, **kwargs)


class CReLU(nn.Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(CReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        r = F.relu(input[:, 0], inplace=self.inplace)
        i = F.relu(input[:, 1], inplace=self.inplace)
        arg = ((r > 0) & (i > 0)).int()
        return torch.stack((r * arg, i * arg), dim=1)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
