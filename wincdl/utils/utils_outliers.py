import torch
import torch.nn.functional as F


def check_thresholds(thresholds):
    try:
        lower_threshold, upper_threshold = thresholds
    except ValueError:
        raise ValueError(f"Thresholds should be a tuple but is {type(thresholds)}")

    # Ensure that both thresholds are either float or None
    if lower_threshold is not None and not isinstance(lower_threshold, (float, int)):
        raise ValueError(
            f"Lower threshold should be a float or None but is {type(lower_threshold)}"
        )
    if upper_threshold is not None and not isinstance(upper_threshold, (float, int)):
        raise ValueError(
            f"Upper threshold should be a float or None but is {type(upper_threshold)}"
        )

    # Ensure that if both thresholds are not None, upper is bigger than lower
    if lower_threshold is not None and upper_threshold is not None:
        if lower_threshold >= upper_threshold:
            raise ValueError(
                f"Lower threshold should be smaller than upper threshold but is {lower_threshold} and {upper_threshold}"
            )


def get_thresholds(data, method="quantile", alpha=0.05):
    """
    Compute threshold(s).

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Data vector of shape (batch_size,).
    method : str, optional
        Method for outlier detection.
        - 'quantile' (default): Outliers are determined by values outside the
        specified quantile range.
        - 'quantile_unilateral': Outliers are determined by values above the
        specified quantile threshold.
        - 'iqr': Outliers are determined by values outside the whiskers of the
        interquartile range.
        - 'iqr_unilateral': Outliers are determined by values above the upper
        whisker of the interquartile range.
        - 'zscore': Outliers are determined by values that are a certain
        number of standard deviations away from the mean.
        - 'mad': Outliers are determined by values that are a
        certain number of median absolute deviations away from the median.
        Source: B Iglewicz and DC Hoaglin, How to detect and handle outliers,
        1993, p. 11
    alpha : float, optional (default: 0.05)
        Quantile level or threshold value for outlier detection.
        If the method is 'quantile' or 'quantile_unilateral', alpha is the
        quantile level.
        If the method is 'iqr' or 'iqr_unilateral', alpha is the number of
        interquartile ranges to use.
        If the method is 'zscore' or 'mad', alpha is the number
        of standard deviations to use.

    Returns
    -------
    tuple of float
        Outlier threshold(s). If the method is bilateral, returns (lower_threshold, upper_threshold).
        If the method is unilateral, returns (threshold,).

    Raises
    ------
    ValueError
        If an invalid method is provided.

    Notes
    -----
    Detailed information about each method:
    - 'quantile': The lower and upper thresholds are determined by the specified quantile levels.
    - 'quantile_unilateral': Only the upper threshold is determined by the specified quantile level.
    - 'iqr': The thresholds are determined based on the whiskers of the interquartile range.
    - 'iqr_unilateral': Only the upper threshold is determined based on the upper whisker of the interquartile range.
    - 'zscore': The thresholds are determined based on the mean and standard deviation. A value is considered an outlier if it's
      alpha standard deviations away from the mean.
    - 'mad': The thresholds are determined based on the median and median absolute deviation. A value is considered
      an outlier if it's alpha median absolute deviations away from the median.

    """
    # Check which method to use for outlier detection
    if method == "quantile":
        # Method of quantile bilateral
        # Calculate lower and upper thresholds using quantiles
        lower_threshold = torch.quantile(data, alpha)
        upper_threshold = torch.quantile(data, 1 - alpha)
        thresholds = (lower_threshold.item(), upper_threshold.item())

    elif method == "quantile_unilateral":
        # Method of quantile unilateral
        # Calculate upper threshold using quantile
        upper_threshold = torch.quantile(data, 1 - alpha)
        thresholds = (None, upper_threshold.item())

    elif method == "iqr":
        # Method of interquartile range bilateral
        # Calculate interquartile range and thresholds
        q1 = torch.quantile(data, 0.25)
        q3 = torch.quantile(data, 0.75)
        iqr = q3 - q1
        lower_threshold = q1 - alpha * iqr
        upper_threshold = q3 + alpha * iqr
        thresholds = (lower_threshold.item(), upper_threshold.item())

    elif method == "iqr_unilateral":
        # Method of interquartile range unilateral
        # Calculate interquartile range and upper threshold
        q1 = torch.quantile(data, 0.25)
        q3 = torch.quantile(data, 0.75)
        iqr = q3 - q1
        upper_threshold = q3 + 1.5 * iqr
        thresholds = (None, upper_threshold.item())

    elif method == "zscore":
        # Method of standard deviation
        mean = torch.mean(data)
        std = torch.std(data)
        # Calculate lower and upper thresholds
        upper_threshold = mean + alpha * std
        lower_threshold = mean - alpha * std
        thresholds = (lower_threshold.item(), upper_threshold.item())

    elif method == "mad":
        # Method of Modified Z-score
        median = torch.median(data)
        mad = torch.median(torch.abs(data - median))  # median absolute deviation
        # Scaling factor
        constant = 0.6745
        # "The constant 0.6745 is needed because E(MAD) = 0.6745CT for large n", Iglewicz and Hoaglin, 1993
        # Calculate lower and upper thresholds
        upper_threshold = median + alpha * mad / constant
        lower_threshold = median - alpha * mad / constant
        thresholds = (lower_threshold.item(), upper_threshold.item())

    else:
        # Raise an error if an unsupported method is chosen
        raise ValueError(
            f"Invalid method: {method}, must be one of 'quantile', 'quantile_unilateral', 'iqr', 'iqr_unilateral', 'zscore', or 'mad'"
        )

    check_thresholds(thresholds)

    return thresholds


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
    thresholds=None,
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

    if thresholds is None:
        alpha = kwargs.get("alpha", 0.05)
        thresholds = get_thresholds(data, method=method, alpha=alpha)
    elif isinstance(thresholds, float):
        thresholds = (None, thresholds)

    check_thresholds(thresholds)

    # lower_threshold, upper_threshold = thresholds
    # if upper_threshold is None:
    #     upper_threshold = lower_threshold
    #     outliers_mask = data > upper_threshold
    # else:
    #     outliers_mask = (data < lower_threshold) | (data > upper_threshold)

    # Only take upper threshold into account
    upper_threshold = thresholds[1]

    outliers_mask = data > upper_threshold

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
    thresholds=None,
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
        thresholds=thresholds,
        method=method,
        opening_window=opening_window,
        moving_average=moving_average,
        n_channels=n_channels,
        **kwargs,
    )

    return torch.masked_select(data, ~outliers_mask), outliers_mask


def compute_error(
    prediction,
    X,
    z_hat=None,
    lmbd=1,
    loss_fn=torch.nn.MSELoss(),
    per_patch=True,
    keep_dim=True,
    device="cuda:0",
):
    """Compute (lasso) reconstruction error per patch.

    prediction, X : numpy 3d-array
        (n_trials, n_channels, n_times)

    loss_fn : torch loss function

    per_patch : False or int

    """
    assert (
        prediction.shape == X.shape
    ), f"prediction.shape: {prediction.shape}, X.shape: {X.shape}"

    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction, dtype=torch.float, device=device)

    loss_fn_name = loss_fn.__class__.__name__
    list_loss_fn = ["MSELoss", "L1Loss"]
    assert loss_fn_name in list_loss_fn

    loss_fn = loss_fn.__class__(reduction='none')
    diff = loss_fn(prediction, X)

    if per_patch:
        if isinstance(per_patch, int):
            L = per_patch
        elif isinstance(per_patch, bool):
            if z_hat is not None:
                n_trials, n_atoms, n_times_valid = z_hat.shape
                n_times = X.shape[-1]
                L = n_times - n_times_valid + 1
            else:
                raise ValueError("per_patch is True but z_hat is None")

        # Create a tensor to count the number of valid (non-padded) elements in each sum
        valid_counts = torch.ones_like(diff)

        # Extend on left
        diff = F.pad(diff, (L - 1, 0), "constant", 0)
        valid_counts = F.pad(valid_counts, (L - 1, 0), "constant", 0)
        # Create structuring element (kernel)
        n_channels = X.shape[1]
        se = torch.ones(n_channels, 1, L, device=diff.device)

        # Performing the convolution operation
        diff = F.conv1d(diff.float(), se, padding=0, groups=diff.size(1))
        valid_counts = F.conv1d(
            valid_counts, se, padding=0, groups=valid_counts.size(1)
        )

        # Normalize the convolution output by the count of valid elements
        diff = diff / valid_counts

        assert diff.shape == X.shape, f"diff.shape: {diff.shape}, X.shape: {X.shape}"

        # Sum across channels
        diff = torch.sum(diff, dim=1)

    if z_hat is not None:
        n_trials, n_atoms, n_times_valid = z_hat.shape
        n_times = X.shape[-1]
        L = n_times - n_times_valid + 1

        z_hat = lmbd * torch.abs(z_hat)

        # Create a tensor to count the number of valid (non-padded) elements in each sum
        valid_counts = torch.ones_like(z_hat)

        # Expand z_hat with 0 to match X's shape, plus avoiding border effects
        z_hat = F.pad(z_hat, (L - 1, L - 1), "constant", 0)
        valid_counts = F.pad(valid_counts, (L - 1, L - 1), "constant", 0)

        # Performing the convolution operation
        se = torch.ones(n_atoms, 1, L, device=z_hat.device)
        z_convolved = F.conv1d(z_hat.float(), se, padding=0, groups=z_hat.size(1))
        valid_counts = F.conv1d(
            valid_counts, se, padding=0, groups=valid_counts.size(1)
        )

        # Normalize the convolution output by the count of valid elements
        z_convolved = z_convolved / valid_counts

        assert (
            z_convolved.shape[-1] == n_times
        ), f"z_convolved.shape: {z_convolved.shape}, n_times: {n_times}"

        # Sum across atoms
        z_convolved = torch.sum(z_convolved, dim=1)

        if not per_patch:
            # Duplicate across channels
            z_convolved = z_convolved.unsqueeze(1).expand_as(X)
        # else:
        #     z_convolved /= L

        # Add the values to diff
        diff += z_convolved

    if keep_dim:
        return diff
    else:
        return torch.mean(diff)
