import numpy as np
import torch


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
    overlap: bool, optional
        If True, the subwindows are drawn uniformly from the data. If False, the
        data is split into fixed non-overlapping subwindows.
    device, dtype: str, optional
        Device and data type for the data. If None, the data will be converted to
        torch.tensor with default values.

    """

    def __init__(self, data, sample_window=None, overlap=True, device=None, dtype=None):
        super().__init__()
        assert data.ndim in [3, 4], (
            "Data should be of shape (n_trials, n_channels, *support) with "
            f"support being either 1D or 2D. Got {data.shape=}"
        )
        self.data = data
        self.dimN = 1 if data.ndim == 3 else 2
        self.overlap = overlap

        self.n_trials = len(data)

        self.device = device
        self.dtype = dtype

        # Validate sample_window
        if sample_window is None:
            sample_window = data.shape[2:]
            self.data = data.clone().detach().to(device=self.device, dtype=self.dtype)
        elif isinstance(sample_window, int):
            sample_window = tuple(sample_window for _ in range(self.dimN))
        assert len(sample_window) == self.dimN, (
            "sample_window should either be a int or a tuple matching the data "
            f"dimensionality. Got {sample_window=} for {self.dimN}D data."
        )

        # make sure the sample window is smaller than the data
        self.sample_window = tuple(
            min(w, ds) for w, ds in zip(sample_window, data.shape[2:], strict=False)
        )
        # compute the number of windows. Even when overlap is True, only consider
        # the same number of windows per epoch as if overlap was False.
        self.n_windows = tuple(
            n // sw for n, sw in zip(data.shape[2:], self.sample_window, strict=False)
        )
        if overlap:
            self._shape_windows = tuple(
                ds - sw + 1 for ds, sw in zip(data.shape[2:], self.sample_window, strict=False)
            )
        else:
            self._shape_windows = self.n_windows

        self._shape_windows = (self.n_trials, *self._shape_windows)
        self.n_windows = int(np.prod((self.n_trials, *self.n_windows)))

    def __getitem__(self, idx):
        # Adding support for negative indexing
        if idx < 0:
            idx += len(self)

        # Using unravel_index to get the sample index and the window indices
        idx_samp, *idx_windows = np.unravel_index(idx, self._shape_windows)
        slice_window = [
            slice(i, i + sw) if self.overlap else slice(i * sw, i * sw + sw)
            for i, sw in zip(idx_windows, self.sample_window, strict=False)
        ]

        return (
            self.data[(idx_samp, slice(None), *slice_window)]
            .clone()
            .detach()
            .to(device=self.device, dtype=self.dtype)
        )

    def __len__(self):
        return np.prod(self._shape_windows)
