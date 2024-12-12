import numpy as np
import torch
from torch.utils.data import Dataset


# ======== CUSTOM DATASET ========
class SignalDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ======== ANOMALY DETECTION FILTERS ========
def filter_percentile(windows, data, percentile):
    """Filter windows using percentile method."""
    normal_windows = windows.copy()
    for feature in range(data.shape[1]):
        upper = np.percentile(data[:, feature, :], percentile)
        normal_windows = normal_windows[
            # If any value in the window is above the upper percentile,
            # the window is considered an anomaly and is removed
            ~(normal_windows[:, feature, :] >= upper).any(axis=1)
        ]
    return normal_windows


def filter_iqr(windows, data, k):
    """Filter windows using IQR method."""
    normal_windows = windows.copy()
    for feature in range(data.shape[1]):
        q1 = np.percentile(data[:, feature], 25)
        q3 = np.percentile(data[:, feature], 75)
        iqr = q3 - q1
        upper = q3 + k * iqr
        normal_windows = normal_windows[
            (normal_windows[:, feature, :] <= upper).all(axis=1)
        ]
    return normal_windows


def filter_zscore(windows, data, threshold):
    """Filter windows using Z-Score method."""
    normal_windows = windows.copy()
    for feature in range(data.shape[1]):
        mean = np.mean(data[:, feature])
        std = np.std(data[:, feature])
        upper = mean + threshold * std
        normal_windows = normal_windows[
            (normal_windows[:, feature, :] <= upper).all(axis=1)
        ]
    return normal_windows


def filter_mad(windows, data, threshold):
    """Filter windows using MAD method."""
    normal_windows = windows.copy()
    for feature in range(data.shape[1]):
        median = np.median(data[:, feature])
        mad = np.median(np.abs(data[:, feature] - median))
        upper = median + threshold * mad
        normal_windows = normal_windows[
            (normal_windows[:, feature, :] <= upper).all(axis=1)
        ]
    return normal_windows
