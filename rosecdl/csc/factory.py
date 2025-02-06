import torch

from rosecdl.csc.base import ConvolutionalSparseCoder
from rosecdl.csc.dim1 import CSC1d, Rank1CSC1d
from rosecdl.csc.dim2 import CSC2d


def csc_factory(
    lmbd: float,
    n_components: int,
    kernel_size: tuple,
    n_channels: int,
    D_init,
    rank1: bool,
    window: bool,
    positive_D: bool,
    positive_z: bool,
    n_iterations: int,
    deepcdl: bool,
    random_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ConvolutionalSparseCoder:
    """Instantiate a Convolutional Sparse (en)Coder."""
    signal_dimension = len(kernel_size)

    if signal_dimension == 1:
        csc_class = Rank1CSC1d if rank1 else CSC1d
    elif signal_dimension == 2:
        if rank1:
            msg = "Rank1 is only possible for 1d CSC"
            raise ValueError(msg)
        csc_class = CSC2d
    else:
        msg = f"CSC in dim {signal_dimension} is not implemented"
        raise NotImplementedError(msg)

    return csc_class(
        lmbd=lmbd,
        n_components=n_components,
        kernel_size=kernel_size,
        n_channels=n_channels,
        D_init=D_init,
        window=window,
        positive_D=positive_D,
        positive_z=positive_z,
        n_iterations=n_iterations,
        deepcdl=deepcdl,
        random_state=random_state,
        device=device,
        dtype=dtype,
    )
