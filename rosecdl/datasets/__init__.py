import numpy as np
import torch

from rosecdl.datasets.subwindow_dataset import SubwindowsDataset
from rosecdl.utils.utils import get_torch_generator


def create_dataloader(
    data,
    mini_batch_size=10,
    sample_window=None,
    overlap=True,
    random_state=None,
    device=None,
    dtype=None,
    **kwargs_dataset,
):
    """Create dataset for conv signals.

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
    generator = get_torch_generator(random_state)
    if isinstance(data, np.ndarray | torch.Tensor):
        dataset = SubwindowsDataset(
            data,
            sample_window=sample_window,
            overlap=overlap,
            device=device,
            dtype=dtype,
        )
    elif data == "physionet":
        from rosecdl.physionet import PhysionetDataset

        dataset = PhysionetDataset(
            **kwargs_dataset, sample_window=sample_window, dtype=dtype, device=device
        )
    elif isinstance(data, str):
        from rosecdl.meg import MEGPopDataset

        dataset = MEGPopDataset(
            data,
            sample_window=sample_window,
            n_samples=kwargs_dataset.get("n_samples", None),
            device=device,
            dtype=dtype,
        )

    sampler = torch.utils.data.RandomSampler(
        dataset,
        num_samples=dataset.n_windows,
        generator=generator,
        # If the dataset length is too high, it takes very long time to sample
        # without replacement with this sampler, but there is a low chance to
        # get issue with replacement
        replacement=len(dataset) > 1e6,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=mini_batch_size, sampler=sampler
    )
