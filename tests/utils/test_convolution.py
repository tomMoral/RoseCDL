import pytest
import torch
from torch.nn.functional import conv1d, conv2d, conv_transpose1d, conv_transpose2d

from rosecdl.utils.convolution import fft_conv, fft_conv_transpose


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("n_atoms", [2, 4, 10])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("signal_size", [10, 20, 50, 100])
@pytest.mark.parametrize("kernel_size", [3, 5, 9])
@pytest.mark.parametrize("seed", [40, 41, 42])
def test_fft_conv1d(batch_size, n_channels, signal_size, n_atoms, kernel_size, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)

    x = torch.rand(batch_size, n_channels, signal_size, generator=rng)
    D = torch.rand(n_atoms, n_channels, kernel_size, generator=rng)

    fft_res = fft_conv(x, D)
    torch_res = conv1d(x, D)

    assert torch.allclose(fft_res, torch_res)


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("n_atoms", [2, 4, 10])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("signal_size", [10, 20, 50, 100])
@pytest.mark.parametrize("kernel_size", [3, 5, 9])
@pytest.mark.parametrize("seed", [40, 41, 42])
def test_fft_conv_transpose1d(
    batch_size, n_channels, signal_size, n_atoms, kernel_size, seed
):
    rng = torch.Generator()
    rng.manual_seed(seed)

    z = torch.rand(batch_size, n_atoms, signal_size, generator=rng)
    D = torch.rand(n_atoms, n_channels, kernel_size, generator=rng)

    fft_res = fft_conv_transpose(z, D)
    torch_res = conv_transpose1d(z, D)

    atol = 1e-6 * torch.abs(torch_res).max()

    assert torch.allclose(fft_res, torch_res, atol=atol)


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("n_atoms", [2, 4, 10])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("signal_size", [(20, 20), (50, 50), (100, 100)])
@pytest.mark.parametrize("kernel_size", [(5, 5), (9, 9)])
@pytest.mark.parametrize("seed", [41, 42])
def test_fft_conv2d_batch(
    batch_size, n_channels, signal_size, n_atoms, kernel_size, seed
):
    rng = torch.Generator()
    rng.manual_seed(seed)

    x = torch.rand(batch_size, n_channels, *signal_size, generator=rng)
    D = torch.rand(n_atoms, n_channels, *kernel_size, generator=rng)

    fft_res = fft_conv(x, D)
    torch_res = conv2d(x, D)

    assert torch.allclose(fft_res, torch_res)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_atoms", [4])
@pytest.mark.parametrize("n_channels", [3])
@pytest.mark.parametrize("signal_rows", [10, 19, 50])
@pytest.mark.parametrize("signal_cols", [10, 29, 50])
@pytest.mark.parametrize("kernel_rows", [4, 6, 9])
@pytest.mark.parametrize("kernel_cols", [3, 6, 9])
@pytest.mark.parametrize("seed", [42])
def test_fft_conv2d_shapes(
    batch_size,
    n_channels,
    signal_rows,
    signal_cols,
    n_atoms,
    kernel_rows,
    kernel_cols,
    seed,
):
    rng = torch.Generator()
    rng.manual_seed(seed)

    x = torch.rand(batch_size, n_channels, signal_rows, signal_cols, generator=rng)
    D = torch.rand(n_atoms, n_channels, kernel_rows, kernel_cols, generator=rng)

    fft_res = fft_conv(x, D)
    torch_res = conv2d(x, D)

    assert torch.allclose(fft_res, torch_res)


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("n_atoms", [4, 10])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("signal_size", [(20, 20), (50, 50), (100, 100)])
@pytest.mark.parametrize("kernel_size", [(5, 5), (9, 9)])
@pytest.mark.parametrize("seed", [41, 42])
def test_fft_conv_transpose2d_batch(
    batch_size, n_channels, signal_size, n_atoms, kernel_size, seed
):
    rng = torch.Generator()
    rng.manual_seed(seed)

    z = torch.rand(batch_size, n_atoms, *signal_size, generator=rng)
    D = torch.rand(n_atoms, n_channels, *kernel_size, generator=rng)

    fft_res = fft_conv_transpose(z, D)
    torch_res = conv_transpose2d(z, D)

    atol = 1e-6 * torch.abs(torch_res).max()

    assert torch.allclose(fft_res, torch_res, atol=atol)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_atoms", [4])
@pytest.mark.parametrize("n_channels", [3])
@pytest.mark.parametrize("signal_rows", [10, 19, 50])
@pytest.mark.parametrize("signal_cols", [10, 29, 50])
@pytest.mark.parametrize("kernel_rows", [4, 6, 9])
@pytest.mark.parametrize("kernel_cols", [3, 6, 9])
@pytest.mark.parametrize("seed", [42])
def test_fft_conv_transpose2d_shapes(
    batch_size,
    n_channels,
    signal_rows,
    signal_cols,
    n_atoms,
    kernel_rows,
    kernel_cols,
    seed,
):
    rng = torch.Generator()
    rng.manual_seed(seed)

    z = torch.rand(batch_size, n_atoms, signal_rows, signal_cols, generator=rng)
    D = torch.rand(n_atoms, n_channels, kernel_rows, kernel_cols, generator=rng)

    fft_res = fft_conv_transpose(z, D)
    torch_res = conv_transpose2d(z, D)

    atol = 1e-6 * torch.abs(torch_res).max()

    assert torch.allclose(fft_res, torch_res, atol=atol)
