import numpy as np
import pytest
import torch

from rosecdl.datasets.subwindow_dataset import SubwindowsDataset


class TestSubwindowsDataset:
    @pytest.mark.parametrize("n_trials", [1, 2])
    @pytest.mark.parametrize("sample_window", [None, 5])
    def test_1d_no_window(self, n_trials, sample_window):
        n_times = 10
        data = torch.zeros(n_trials, 1, n_times)
        dataset = SubwindowsDataset(data, sample_window=sample_window)

        n_windows = n_times - sample_window + 1 if sample_window is not None else 1
        expected_len = n_trials if sample_window is None else n_trials * n_windows
        assert len(dataset) == expected_len

        if sample_window is not None:
            for i in range(len(dataset)):
                assert dataset[i].shape == (1, sample_window)

    @pytest.mark.parametrize("n_trials", [1, 2])
    @pytest.mark.parametrize("sample_window", [None, (5, 4)])
    def test_2d_no_window(self, n_trials, sample_window):
        support = (10, 11)
        data = torch.zeros(n_trials, 1, *support)
        dataset = SubwindowsDataset(data, sample_window=sample_window)

        n_windows = np.prod(
            tuple(s - sw + 1 for s, sw in zip(support, sample_window, strict=True))
            if sample_window is not None
            else 1
        )
        expected_len = n_trials if sample_window is None else n_trials * n_windows
        assert len(dataset) == expected_len

        if sample_window is not None:
            for i in range(len(dataset)):
                assert dataset[i].shape == (1, *sample_window)
