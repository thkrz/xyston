from torch.nn.common_types import _size_4_t, _size_6_t


def _div(numerator: int, denominator: int, ceil_mode: bool) -> int:
    if ceil_mode:
        return -(-numerator // denominator)
    return numerator // denominator


def _output_size_4_t(
    size: _size_6_t,
    padding: _size_4_t,
    dilation: _size_4_t,
    kernel_size: _size_4_t,
    stride: _size_4_t,
    ceil_mode: bool = False,
) -> _size_4_t:
    shape = list(size[:2])
    size = size[2:]
    shape += [
        _output_size(
            size[i], padding[i], dilation[i], kernel_size[i], stride[i], ceil_mode
        )
        for i in range(4)
    ]
    return tuple(shape)


def _output_size(
    size: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    stride: int,
    ceil_mode: bool = False,
) -> int:
    if padding is None:
        return size
    num = size + 2 * padding - dilation * (kernel_size - 1) - 1
    return _div(num, stride, ceil_mode) + 1


def _pooling_size_4_t(
    size: _size_6_t,
    padding: _size_4_t,
    dilation: _size_4_t,
    kernel_size: _size_4_t,
    stride: _size_4_t,
    ceil_mode: bool,
) -> _size_4_t:
    shape = list(size[:2])
    size = size[2:]
    shape += [
        _pooling_size(size[i], padding[i], kernel_size[i], stride[i], ceil_mode)
        for i in range(4)
    ]
    return tuple(shape)


def _pooling_size(
    size: int, padding: int, kernel_size: int, stride: int, ceil_mode: bool
) -> int:
    num = size + 2 * padding - kernel_size
    return _div(num, stride, ceil_mode) + 1
