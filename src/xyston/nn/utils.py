import torch
from torch.nn.common_types import _size_4_t, _size_any_t


def _div(numerator: int, denominator: int, ceil_mode: bool) -> int:
    if ceil_mode:
        return -(-numerator // denominator)
    return numerator // denominator


def output_shape(
    size: _size_any_t,
    padding: _size_4_t,
    dilation: _size_4_t,
    kernel_size: _size_4_t,
    stride: _size_4_t,
    ceil_mode: bool = False,
) -> _size_4_t:
    def shape(
        size: int,
        padding: int,
        dilation: int,
        kernel_size: int,
        stride: int,
        ceil_mode: bool,
    ) -> int:
        num = size + 2 * padding - dilation * (kernel_size - 1) - 1
        return _div(num, stride, ceil_mode) + 1

    if padding is None:
        return size
    if not isinstance(size, tuple):
        return shape(size, padding, dilation, kernel_size, stride, ceil_mode)
    s = list(size[:2])
    size = size[2:]
    return tuple(
        s
        + [
            shape(
                size[i],
                padding[i],
                dilation[i],
                kernel_size[i],
                stride[i],
                ceil_mode,
            )
            for i in range(4)
        ]
    )


def pooling_output_shape(
    size: _size_any_t,
    padding: _size_4_t,
    kernel_size: _size_4_t,
    stride: _size_4_t,
    ceil_mode: bool = False,
) -> _size_4_t:
    def shape(
        size: int, padding: int, kernel_size: int, stride: int, ceil_mode: bool
    ) -> int:
        num = size + 2 * padding - kernel_size
        return _div(num, stride, ceil_mode) + 1

    if not isinstance(size, tuple):
        return shape(size, padding, kernel_size, stride, ceil_mode)
    s = list(size[:2])
    size = size[2:]
    return tuple(
        s
        + [
            shape(size[i], padding[i], kernel_size[i], stride[i], ceil_mode)
            for i in range(4)
        ]
    )


def zeros_like(input, shape=None, dtype=None):
    dtype = input.dtype if dtype is None else dtype
    shape = input.shape if shape is None else shape
    return torch.zeros(
        shape,
        dtype=dtype,
        layout=input.layout,
        device=input.device,
    )
