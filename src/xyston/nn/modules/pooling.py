import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_4_t
from torch.nn.modules.pooling import _AvgPoolNd
from torch.nn.modules.utils import _quadruple

from .. import functional as F


class AvgPool4d(_AvgPoolNd):
    kernel_size: _size_4_t
    stride: _size_4_t
    padding: _size_4_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_4_t,
        stride: _size_4_t = None,
        padding: _size_4_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super(AvgPool4d, self).__init__()
        self.kernel_size = _quadruple(kernel_size)
        self.stride = _quadruple(stride if stride is not None else kernel_size)
        self.padding = _quadruple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool4d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class _CAvgPoolNd(nn.Module):
    def __init__(self, cls, *args, **kwargs):
        super(_CAvgPoolNd, self).__init__()
        self.pooling = cls(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        r = self.pooling(input[:, 0])
        i = self.pooling(input[:, 1])
        return torch.stack((r, i), dim=1)


def CAvgPool1d(*args, **kwargs):
    return _CAvgPoolNd(nn.AvgPool1d, *args, **kwargs)


def CAvgPool2d(*args, **kwargs):
    return _CAvgPoolNd(nn.AvgPool2d, *args, **kwargs)


def CAvgPool3d(*args, **kwargs):
    return _CAvgPoolNd(nn.AvgPool3d, *args, **kwargs)


def CAvgPool4d(*args, **kwargs):
    return _CAvgPoolNd(AvgPool4d, *args, **kwargs)
