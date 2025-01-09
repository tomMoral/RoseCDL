import torch
import torch.nn.functional as F


def check_threshold(threshold):
    # Ensure that the threshold is either float or None
    if threshold is not None and not isinstance(threshold, (float, int)):
        raise ValueError(
            f"threshold should be a float or None but is {type(threshold)}"
        )



def get_threshold(data, method="quantile", alpha=0.05):
    """
    Compute the outlier detection threshold.

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Data vector of shape (batch_size,).
    method : str, optional
        Method for outlier detection.
        - 'quantile' (default): Outliers are determined by values outside the
        specified quantile range.
        - 'iqr': Outliers are determined by values outside the whiskers of the
        interquartile range.
        - 'zscore': Outliers are determined by values that are a certain
        number of standard deviations away from the mean.
        - 'mad': Outliers are determined by values that are a
        certain number of median absolute deviations away from the median.
    alpha : float, optional (default: 0.05)
        Quantile level or threshold value for outlier detection.
        If the method is 'quantile', alpha is the quantile level.
        If the method is 'iqr', alpha is the number of interquartile ranges to use.
        If the method is 'zscore' or 'mad', alpha is the number of standard
        deviations to use.

    Returns
    -------
    float
        Outlier threshold.

    Raises
    ------
    ValueError
        If an invalid method is provided.

    Notes
    -----
    Detailed information about each method:
    - 'quantile': The lower and upper threshold are determined by the specified quantile levels.
    - 'quantile_unilateral': Only the upper threshold is determined by the specified quantile level.
    - 'iqr': The threshold are determined based on the whiskers of the interquartile range.
    - 'iqr_unilateral': Only the upper threshold is determined based on the upper whisker of the interquartile range.
    - 'zscore': The threshold are determined based on the mean and standard deviation. A value is considered an outlier if it's
      alpha standard deviations away from the mean.
    - 'mad': The threshold are determined based on the median and median absolute deviation. A value is considered
      an outlier if it's alpha median absolute deviations away from the median.

    """
    # Check which method to use for outlier detection
    if method == "quantile":
        # Method of quantile bilateral
        # Calculate lower and upper threshold using quantiles
        threshold = torch.quantile(data, 1 - alpha)

    elif method == "iqr":
        # Method of interquartile range bilateral
        # Calculate interquartile range and threshold
        q1 = torch.quantile(data, 0.25)
        q3 = torch.quantile(data, 0.75)
        iqr = q3 - q1
        threshold = q3 + alpha * iqr

    elif method == "zscore":
        # Method of standard deviation
        mean = torch.mean(data)
        std = torch.std(data)
        # Calculate lower and upper threshold
        threshold = mean + alpha * std

    elif method == "mad":
        # Method of Modified Z-score
        median = torch.median(data)
        mad = torch.median(torch.abs(data - median))  # median absolute deviation
        # Scaling factor
        constant = 0.6745
        # "The constant 0.6745 is needed because E(MAD) = 0.6745CT for large n", Iglewicz and Hoaglin, 1993
        # Calculate lower and upper threshold
        threshold = median + alpha * mad / constant

    else:
        # Raise an error if an unsupported method is chosen
        raise ValueError(
            f"Invalid method: {method}, must be one of 'quantile', 'quantile_unilateral', 'iqr', 'iqr_unilateral', 'zscore', or 'mad'"
        )

    threshold = threshold.item()
    check_threshold(threshold)

    return threshold


def gaussian_kernel(size, sigma):
    """Generates a 1D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
    return gauss / gauss.sum()


def apply_moving_average(data, window_size=15, method="average"):
    """

    Parameters
    ----------
    data : torch.Tensor
        Data vector of shape (batch_size, n_channels, n_times).

    window_size : int, optional
        Size of the window, by default 15

    method : str, optional
        Method for moving average. Can be 'average', 'max', or 'gaussian', by default 'average'

    Returns
    -------
    torch.Tensor
        Data vector of shape (batch_size, n_channels, n_times) after moving average.

    """
    if window_size % 2 != 1:
        window_size += 1

    n_dims = data.dim()
    original_shape = data.shape  # Store the original shape

    # Reshape data for 1D convolution
    if n_dims == 2:  # Shape (n_trials, n_times)
        data = data.unsqueeze(1)  # Add a channel dimension
    elif n_dims != 3:  # Shape (n_trials, n_channels, n_times)
        raise ValueError("Data must be either 2D or 3D.")

    if method == "average":
        # Create structuring element (kernel)
        se = torch.ones(data.size(1), 1, window_size, device=data.device) / window_size
        # Perform convolution
        convolved_data = F.conv1d(
            data.float(), se, padding=window_size // 2, groups=data.size(1)
        )
    elif method == "gaussian":
        sigma = window_size / 3  # A common choice for sigma relative to window size
        se = gaussian_kernel(window_size, sigma).to(data.device).view(1, 1, -1)
        se = se.repeat(data.size(1), 1, 1)  # Repeat kernel
        convolved_data = F.conv1d(
            data.float(), se, padding=window_size // 2, groups=data.size(1)
        )
    elif method == "max":
        # Perform max pooling
        convolved_data = F.max_pool1d(
            data.float(), window_size, stride=1, padding=window_size // 2
        )

    # Reshape back to original shape
    if n_dims == 2:
        convolved_data = convolved_data.squeeze(1)

    assert (
        convolved_data.shape == original_shape
    ), f"Shape mismatch: {convolved_data.shape}, {original_shape}"
    return convolved_data


def apply_opening(outliers_mask, window_size=15):
    """ """
    # Apply opening to remove isolated outliers, see Mathematical morphology
    # Ensure it is a strictly positive int
    window_size = int(window_size)
    if window_size <= 0:
        # Do nothing
        return outliers_mask

    ndim = outliers_mask.ndim
    original_shape = outliers_mask.shape

    if ndim == 2:
        outliers_mask = outliers_mask.unsqueeze(1)
        # n_channels = 1
    elif ndim == 3:
        # n_channels = outliers_mask.shape[1]
        pass
    else:
        raise ValueError(f"outliers_mask should be 2D or 3D but is {ndim}D")

    # Pad mask to avoid boundary effects
    outliers_mask = F.pad(outliers_mask.float(), (0, window_size - 1), "constant", 0)
    # Creating a structuring element for the convolution
    se = torch.ones(outliers_mask.size(1), 1, window_size, device=outliers_mask.device)
    # Performing the convolution operation
    convolved_mask = F.conv1d(
        outliers_mask.float(), se, padding=0, groups=outliers_mask.size(1)
    )
    # Thresholding to get the final mask
    convolved_outliers_mask = (convolved_mask > 0).bool()

    if ndim == 2:
        convolved_outliers_mask = convolved_outliers_mask.squeeze(1)

    assert (
        convolved_outliers_mask.shape == original_shape
    ), f"convolved_outliers_mask.shape: {convolved_outliers_mask.shape}, outliers_mask.shape: {original_shape}"

    return convolved_outliers_mask


def get_outlier_mask(
    data,
    threshold=None,
    method="quantile",
    moving_average=None,
    opening_window=None,
    union_channels=True,
    **kwargs,
):
    if moving_average is not None:
        if not isinstance(moving_average, dict):
            moving_average = {}  # Apply with default parameters
        data = apply_moving_average(data, **moving_average)

    if threshold is None:
        alpha = kwargs.get("alpha", 0.05)
        threshold = get_threshold(data, method=method, alpha=alpha)

    check_threshold(threshold)

    # Compute the mask to detect the outliers
    outliers_mask = data > threshold

    if opening_window is not None:
        if not isinstance(opening_window, int):
            opening_window = 15  # Apply with default parameters
        outliers_mask = apply_opening(outliers_mask, window_size=opening_window)

    if outliers_mask.ndim == 3 and union_channels:
        # Take the boolean union "OR" across channels
        outliers_mask = outliers_mask.max(dim=1).values
        outliers_mask = (outliers_mask > 0).bool()
        # Back to 3D
        outliers_mask = outliers_mask.unsqueeze(1)
        # Expanding the mask to match data's shape
        outliers_mask = outliers_mask.expand_as(data)

    assert (
        outliers_mask.shape == data.shape
    ), f"outliers_mask.shape: {outliers_mask.shape}, data.shape: {data.shape}"

    return outliers_mask


def remove_outliers(
    data,
    threshold=None,
    method="quantile",
    opening_window=None,
    moving_average=None,
    n_channels=1,
    **kwargs,
):
    """
    Remove outliers from a batch vector.

    Parameters
    ----------
    moving_average : dict, optional
        Moving average parameters, by default None
        example: moving_average=dict(
            window_size=int(model.n_times_atom),
            method='max',  # 'max' or 'average'
            gaussian=False,
        )

    """

    outliers_mask = get_outlier_mask(
        data,
        threshold=threshold,
        method=method,
        opening_window=opening_window,
        moving_average=moving_average,
        n_channels=n_channels,
        **kwargs,
    )

    return torch.masked_select(data, ~outliers_mask), outliers_mask
