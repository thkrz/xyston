import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.common_types import _size_4_t
from torch.nn.modules.utils import _quadruple
from typing import Optional, Union

from .. import functional as F


class Conv4d(nn.modules.conv._ConvNd):
    __doc__ = r"""Applies a 4D convolution over an input signal composed of several input
    planes.
    """

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
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if padding_mode != "zeros":
            raise ValueError(
                'Only "zeros" padding mode is supported for {}'.format(
                    self.__class__.__name__
                )
            )
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
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return F.conv4d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class _CConvNd(nn.Module):
    def __init__(self, cls, *args, **kwargs) -> None:
        super(_CConvNd, self).__init__()
        kernel_size = tuple(zip(*kwargs["kernel_size"]))
        kwargs["kernel_size"] = kernel_size[0]
        self.conv_r = cls(*args, **kwargs)
        kwargs["kernel_size"] = kernel_size[1]
        self.conv_i = cls(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        xa = self.conv_r(input[:, 0])
        ya = self.conv_r(input[:, 1])
        xb = self.conv_i(input[:, 0])
        yb = self.conv_i(input[:, 1])
        r = xa - yb
        i = xb + ya
        return torch.stack((r, i), dim=1)


def CConv1d(*args, **kwargs):
    return _CConvNd(nn.Conv1d, *args, **kwargs)


def CConv2d(*args, **kwargs):
    return _CConvNd(nn.Conv2d, *args, **kwargs)


def CConv3d(*args, **kwargs):
    return _CConvNd(nn.Conv3d, *args, **kwargs)


def CConv4d(*args, **kwargs):
    return _CConvNd(Conv4d, *args, **kwargs)
