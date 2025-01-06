import numpy as np
from alphacsc.utils.dictionary import _patch_reconstruction_error, get_D_shape, get_uv


def get_z_nnz(z_hat):
    """Calculate the number of non-zero elements across specified axes of z_hat array.

    z_hat : numpy.ndarray
        Input array from which to count non-zero elements. Expected to be a 3D array.

    numpy.ndarray
        Array containing counts of non-zero elements summed across axes 0 and 2,
        preserving the dimension of axis 1.

    Notes
    -----
    The function calculates element-wise count of non-zero values in z_hat
    across the first (0) and last (2) axes, effectively reducing the 3D array
    to a 1D array of counts.
    """
    z_nnz = np.sum(z_hat != 0, axis=(0, 2))
    # return z_nnz / z_nnz.shape[-1]
    return z_nnz


def get_max_error_patch(X, z, D):
    """Find the patch with maximum reconstruction error.

    This function identifies and extracts the patch from the signal that has the
    highest reconstruction error when reconstructed using the given dictionary.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times) or (n_trials, n_channels * n_times)
        The data matrix, either in multivariate form (n_channels, n_times) or
        in univariate form (n_channels * n_times).
    z : ndarray
        The sparse activation coefficients used for signal reconstruction.
    D : ndarray, shape (n_atoms, n_channels * n_times_atom) or (n_atoms, n_channels, n_times_atom)
        The dictionary used for reconstruction. Can be in univariate or
        multivariate form.

    Returns
    -------
    patch : ndarray, shape (1, n_channels, n_times_atom) or (1, n_channels * n_times_atom)
        The extracted patch with maximum reconstruction error. The shape matches
        the input format of D (multivariate or univariate).

    Notes
    -----
    The function computes reconstruction errors for all possible patches in the signal
    and returns the one with maximum error. The returned patch format (univariate or
    multivariate) matches the format of the input dictionary D.
    """

    patch_rec_error = _patch_reconstruction_error(X, z, D)
    i0 = patch_rec_error.argmax()
    n0, t0 = np.unravel_index(i0, patch_rec_error.shape)

    n_channels = X.shape[1]
    *_, n_times_atom = get_D_shape(D, n_channels)

    patch = X[n0, :, t0 : t0 + n_times_atom][None]
    if D.ndim == 2:
        patch = get_uv(patch)
    return patch.copy()


# ======== ANOMALY DETECTION FILTERS ========
def filter_percentile(windows, data, percentile):
    """Filter windows based on percentile thresholding for anomaly detection.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, n_features, window_size)
        The windows to be filtered.
    data : ndarray, shape (n_samples, n_features, window_size)
        The reference data used to compute the percentile threshold.
    percentile : float
        The percentile value (0-100) used as threshold. Values above this
        percentile are considered anomalies.

    Returns
    -------
    ndarray
        Filtered windows with anomalies removed, maintaining the same structure
        as input windows but potentially with fewer samples.
    """
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
    """Filter windows using Interquartile Range (IQR) method for anomaly detection.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, n_features, window_size)
        The windows to be filtered.
    data : ndarray, shape (n_samples, n_features)
        The reference data used to compute IQR thresholds.
    k : float
        The multiplier for IQR to set the threshold. Common values are 1.5 or 3.
        Threshold is set at Q3 + k*IQR.

    Returns
    -------
    ndarray
        Filtered windows with anomalies removed, maintaining the same structure
        as input windows but potentially with fewer samples.

    Notes
    -----
    IQR = Q3 - Q1, where Q1 and Q3 are the 25th and 75th percentiles respectively.
    Windows with any values above Q3 + k*IQR are considered anomalies.
    """
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
    """Filter windows using Z-Score method for anomaly detection.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, n_features, window_size)
        The windows to be filtered.
    data : ndarray, shape (n_samples, n_features)
        The reference data used to compute mean and standard deviation.
    threshold : float
        The number of standard deviations to use as threshold.
        Values above mean + threshold*std are considered anomalies.

    Returns
    -------
    ndarray
        Filtered windows with anomalies removed, maintaining the same structure
        as input windows but potentially with fewer samples.

    Notes
    -----
    Z-score measures how many standard deviations away from the mean a data
    point is. This method assumes normally distributed data.
    """
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
    """Filter windows using Median Absolute Deviation (MAD) method for anomaly detection.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, n_features, window_size)
        The windows to be filtered.
    data : ndarray, shape (n_samples, n_features)
        The reference data used to compute median and MAD.
    threshold : float
        The multiplier for MAD to set the threshold.
        Values above median + threshold*MAD are considered anomalies.

    Returns
    -------
    ndarray
        Filtered windows with anomalies removed, maintaining the same structure
        as input windows but potentially with fewer samples.

    Notes
    -----
    MAD is a robust measure of variability that is more resilient to outliers
    than standard deviation. It's calculated as the median of absolute deviations
    from the median.
    """
    normal_windows = windows.copy()
    for feature in range(data.shape[1]):
        median = np.median(data[:, feature])
        mad = np.median(np.abs(data[:, feature] - median))
        upper = median + threshold * mad
        normal_windows = normal_windows[
            (normal_windows[:, feature, :] <= upper).all(axis=1)
        ]
    return normal_windows
