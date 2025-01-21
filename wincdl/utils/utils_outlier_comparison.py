import numpy as np
import torch

from wincdl.loss import LassoLoss, OutlierLoss


def get_support_mean(data, outlier_mask):
    """Get the mean of the data without the outliers."""
    axis = tuple(range(2, data.ndim))
    support_mean = (data * (~outlier_mask)).sum(axis=axis, keepdims=True)
    support_mean /= np.maximum((~outlier_mask).sum(axis=axis, keepdims=True), 1)
    return support_mean


def remove_outliers_before_cdl(
    data: np.array,
    activation_vector_shape: tuple,
    method: str,
    alpha: float,
    moving_average: int,
    opening_window: bool,
    union_channels: bool,
    fill_by_channel=True,
) -> np.array:
    """Remove outliers before CDL.

    Args:
        data: array of shape (n_trials, n_channels, n_times)
        activation_vector_shape: shape of the activation vector
        method: name of the outlier detection method
        alpha: parameter of the method
        moving_average: size of the moving average window
        opening_window: whether to open the window
        union_channels: whether to use the union of the channels
        fill_by_channel: whether to replace outlier by the mean of the signal without outliers
                         by channels or globally
    """
    # Check data type
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    outlier_loss = OutlierLoss(
        LassoLoss(lmbd=0, reduction="mean"),
        method=method,
        alpha=alpha,
        moving_average=moving_average,
        opening_window=opening_window,
        union_channels=union_channels,
    )

    outlier_mask = (
        outlier_loss.get_outliers_mask(
            X_hat=torch.from_numpy(np.zeros_like(data)),
            z_hat=torch.zeros(activation_vector_shape),
            X=torch.from_numpy(data),
        )
        .cpu()
        .numpy()
    )
    outlier_mask = np.broadcast_to(outlier_mask, data.shape)
    if fill_by_channel:
        support_mean = get_support_mean(data, outlier_mask)
        # Flatten to allow data assignment
        data_clean, outlier_mask = data.ravel().copy(), outlier_mask.ravel()
        support_mean = np.broadcast_to(support_mean, data.shape).ravel()
        data_clean[outlier_mask] = support_mean[outlier_mask]
        return data.reshape(data.shape)
    else:
        data_clean = data.copy()
        data_clean[outlier_mask] = data[~outlier_mask].mean()
        return data_clean
