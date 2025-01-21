import numpy as np
import torch

from ..utils.utils import get_torch_generator
from .subwindow_dataset import SubwindowsDataset


def create_dataloader(
    data,
    mini_batch_size=10,
    sample_window=None,
    overlap=False,
    random_state=None,
    device=None,
    dtype=None,
    **kwargs_dataset,
):
    """
    Create dataset for conv signals

    Parameters
    ----------
    data : str or np.array (n_trials, n_channels, *support)
        Path to data or np.array containing the data.
    device : str
        Device for computations
    dtype : type
        Type of tensors
    mini_batch_size : int, optional
        Size of mini batches, by default 10
    random_state : int, optional
        Seed, by default 2147483647

    Returns
    -------
    torch.utils.data.DataLoader
        Torch DataLoader
    """
    generator = get_torch_generator(random_state, device=device)
    if isinstance(data, np.ndarray):
        dataset = SubwindowsDataset(
            data, sample_window=sample_window, overlap=overlap, device=device, dtype=dtype,
        )
    elif data == "physionet":
        from .physionet import PhysionetDataset
        dataset = PhysionetDataset(
            **kwargs_dataset, window=sample_window,
            dtype=dtype, device=device, seed=(random_state, 1),
        ),
    elif isinstance(data, str):
        from .meg import MEGPopDataset
        dataset = MEGPopDataset(
            data,
            window=sample_window,
            n_samples=kwargs_dataset.get("n_samples", None),
            device=device,
            dtype=dtype,
            seed=(random_state, 1),
        )

    return torch.utils.data.DataLoader(
        dataset, batch_size=mini_batch_size, shuffle=True, generator=generator,
    )
