import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.common_types import _size_4_t
from typing import Optional


def _quadruple(n):
    if type(n) == int:
        return (n,) * 4
    return n


class Conv4d(nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_4_t,
        stride: _size_4_t = 1,
        padding: str = "valid",
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _quadruple(kernel_size)
        stride_ = _quadruple(stride)
        super(Conv4d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding,
            _quadruple(1),
            False,
            _quadruple(0),
            groups,
            bias,
            "zeros",
            **factory_kwargs,
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        b, c_i, l_i, d_i, h_i, w_i = input.shape
        l_k, d_k, h_k, w_k = self.kernel_size
        if self.padding == "valid":
            l_o = l_i - l_k + 1
        else:
            l_o = l_i
        outputs = l_o * [None]
        for i in range(l_k):
            for j in range(0, l_i, self.stride[0]):
                frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if frame < 0 or frame >= l_o:
                    continue
                o = F.conv3d(
                    input[:, :, j, :].view(b, c_i, d_i, h_i, w_i),
                    weight[:, :, i, :, :],
                    bias,
                    self.stride[1:],
                    self.padding,
                    1,
                    self.groups,
                )
                if outputs[frame] is None:
                    outputs[frame] = o
                else:
                    outputs[frame] += o
        return torch.stack(outputs, dim=2)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class LASLNNet(nn.Module):
    def __init__(self):
        super(LASLNNet, self).__init__()
        kernel_size = (3, 3, 3, 3)
        self.conv = nn.Sequential(
            Conv4d(1, 2, kernel_size),
            nn.ReLU(inplace=True),
            Conv4d(2, 4, kernel_size),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(in_features=4 * 3**4, out_features=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return x
        return self.fc(x)


if __name__ == "__main__":
    lnet = LASLNNet()
    print(lnet.forward(None))
