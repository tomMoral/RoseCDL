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
    z = torch.rand(4, 2, 10, 10)
    D = torch.rand(2, 3, 3, 3)

    assert torch.allclose(fft_conv_transpose(z, D), conv_transpose2d(z, D))
