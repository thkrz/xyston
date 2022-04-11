import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple


def _triple(n):
    return (n, n, n)


class Conv4d(nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int, int],
        stride: int = 1,
        padding: str = "VALID",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Conv4d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return F.conv3d(
            input, weight, bias, self.stride, (0, 0, 0, 0), self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        pass
