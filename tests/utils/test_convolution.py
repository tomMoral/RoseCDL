import torch
from torch.nn.functional import conv2d, conv_transpose2d

from rosecdl.utils.convolution import fft_conv, fft_conv_transpose


def test_fft_conv():
    x = torch.rand(3, 2, 10, 10)
    D = torch.rand(4, 2, 3, 3)

    fft_res = fft_conv(x, D)
    torch_res = conv2d(x, D)

    assert torch.allclose(fft_res, torch_res)


def test_fft_conv_transpose():
    z = torch.rand(4, 10, 100, 100)
    D = torch.rand(10, 3, 10, 10)

    fft_res = fft_conv_transpose(z, D)
    torch_res = conv_transpose2d(z, D)

    print(fft_res.min())
    print(fft_res.max())
    print(torch_res.min())
    print(torch_res.max())

    # assert torch.abs(fft_conv_transpose(z, D) - conv_transpose2d(z, D)).max() < 1
