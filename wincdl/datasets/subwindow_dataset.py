import torch
import numpy as np


class SubwindowsDataset(torch.utils.data.Dataset):
    """Dataset to extract subwindows from data.

    This dataset works for in-memory data, either 1D or 2D.

    Parameters
    ----------
    data: np.array, shape (n_trials, n_channels, *support)
        Data to be processed. The data can be 1D or 2D and should have the shape
        (n_trials, n_channels, *support), with the length of `support` being
        the dimensionality. The subwindows will be extracted for each trial, along
        the support dimensions.
    sample_window: int or tuple of int, optional
        Size of minibatch windows. If int, the same window size will be used for
        each dimension. If tuple, the number of elements should match the data
        dimensionality. If None, no subwindows will be extracted and the dataset
        will return each trial as is.
    device, dtype: str, optional
        Device and data type for the data. If None, the data will be converted to
        torch.tensor with default values.
    """

    def __init__(self, data, sample_window=False, device=None, dtype=None):
        super().__init__()
        assert data.ndim in [3, 4], (
            "Data should be of shape (n_trials, n_channels, *support) with "
            f"support being either 1D or 2D. Got {data.shape=}"
        )
        self.data = data
        self.n_samples = data.shape[0]
        self.dimN = 1 if data.ndim == 3 else 2

        self.device = device
        self.dtype = dtype

        self.sto = sample_window is not None
        if sample_window:
            if isinstance(sample_window, int):
                sample_window = tuple(sample_window for _ in range(self.dimN))
            assert len(sample_window) == self.dimN, (
                "sample_window should either be a int or a tuple matching the data "
                f"dimensionality. Got {sample_window=} for {self.dimN}D data."
            )
            self.sample_window = tuple(
                min(w, ds) for w, ds in zip(sample_window, data.shape[2:])
            )
            self.n_windows = tuple(
                ds - sw + 1 for ds, sw in zip(data.shape[2:], self.sample_window)
            )
        else:
            self.n_windows = (1,)
            self.data = torch.tensor(data, device=self.device, dtype=self.dtype)

    def __getitem__(self, idx):
        total_n_windows = np.prod(self.n_windows)
        idx_samp = idx // total_n_windows
        idx = idx % total_n_windows
        if self.sto and self.dimN == 1:
            return torch.tensor(
                self.data[idx_samp, :, idx : (idx + self.sample_window[0])],
                device=self.device,
                dtype=self.dtype,
            )
        elif self.sto and self.dimN == 2:
            idx_i = idx // self.n_windows[1]
            idx_j = idx % self.n_windows[1]
            return torch.tensor(
                self.data[
                    idx_samp, :,
                    idx_i * self.sample_window[0] : (idx_i + 1) * self.sample_window[0],
                    idx_j * self.sample_window[1] : (idx_j + 1) * self.sample_window[1],
                ],
                device=self.device,
                dtype=self.dtype,
            )
        else:
            return self.data[idx_samp]

    def __len__(self):
        if self.sto:
            return self.n_samples * np.prod(self.n_windows)
        else:
            return self.n_samples