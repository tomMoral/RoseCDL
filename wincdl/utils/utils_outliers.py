import torch
import torch.nn.functional as F


def check_threshold(threshold):
    # Ensure that the threshold is either float or None
    if threshold is not None:
        if isinstance(threshold, torch.Tensor):
            assert threshold.dtype in (
                torch.float32,
                torch.int32,
            ), f"threshold.dtype: {threshold.dtype}"
        elif not isinstance(threshold, (float, int)):
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
    dim : int|tuple, optional
        Dimensions along which to compute the threshold. If None, compute the threshold
        over all dimensions.

    Returns
    -------
    threshold : float or Tensor
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


def gaussian_kernel_2d(size, sigma):
    """Generates a 2D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    y = x.unsqueeze(0)
    x = x.unsqueeze(1)
    gauss = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma**2))
    return gauss / gauss.sum()


def apply_moving_average(data, window_size=15, method="average"):
    """Apply moving average to 1D or 2D data.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor of shape:
        - 3D: (batch_size, n_channels, n_times) for 1D signals
        - 4D: (batch_size, n_channels, height, width) for 2D signals
    window_size : int
        Size of the window. For 2D, creates a square window.
    method : str
        Method for moving average:
        - 'average': Simple moving average
        - 'gaussian': Gaussian weighted average
        - 'max': Maximum value in window

    Returns
    -------
    torch.Tensor
        Processed data with same shape as input

    Raises
    ------
    ValueError
        If method is not one of 'average', 'gaussian', or 'max'
        If data dimensions are not 3D or 4D
    """
    # Ensure odd window size
    if window_size % 2 != 1:
        window_size += 1

    ndim = data.dim()
    if ndim not in [3, 4]:
        raise ValueError(f"Data must be 3D or 4D but got {ndim}D")

    if method not in ["average", "gaussian", "max"]:
        raise ValueError(f"Unknown method: {method}")

    original_shape = data.shape
    data = data.float()  # Ensure float type for convolution

    if ndim == 3:  # 1D case
        if method == "average":
            se = (
                torch.ones(data.size(1), 1, window_size, device=data.device)
                / window_size
            )
            result = F.conv1d(data, se, padding=window_size // 2, groups=data.size(1))

        elif method == "gaussian":
            sigma = window_size / 3
            kernel = gaussian_kernel(window_size, sigma).to(data.device).view(1, 1, -1)
            se = kernel.repeat(data.size(1), 1, 1)
            result = F.conv1d(data, se, padding=window_size // 2, groups=data.size(1))

        else:  # method == "max"
            result = F.max_pool1d(data, window_size, stride=1, padding=window_size // 2)

    else:  # 2D case
        if method == "average":
            se = torch.ones(
                data.size(1), 1, window_size, window_size, device=data.device
            ) / (window_size * window_size)
            result = F.conv2d(data, se, padding=window_size // 2, groups=data.size(1))

        elif method == "gaussian":
            sigma = window_size / 3
            kernel = gaussian_kernel_2d(window_size, sigma).to(data.device)
            se = kernel.view(1, 1, window_size, window_size).repeat(
                data.size(1), 1, 1, 1
            )
            result = F.conv2d(data, se, padding=window_size // 2, groups=data.size(1))

        else:  # method == "max"
            result = F.max_pool2d(data, window_size, stride=1, padding=window_size // 2)

    assert (
        result.shape == original_shape
    ), f"Shape mismatch: got {result.shape}, expected {original_shape}"

    return result


def apply_opening(outliers_mask, window_size=15):
    """Apply opening operation to remove isolated outliers.

    Parameters
    ----------
    outliers_mask : torch.Tensor
        Boolean mask of shape:
        - 3D: (batch, channels, n_times) for 1D signals
        - 4D: (batch, channels, height, width) for 2D signals
    window_size : int
        Size of the opening window. For 2D, creates a square window.

    Returns
    -------
    torch.Tensor
        Processed mask with same shape as input
    """
    # Input validation
    if not isinstance(window_size, int) or window_size <= 0:
        return outliers_mask

    ndim = outliers_mask.ndim
    if ndim not in [3, 4]:
        raise ValueError(f"outliers_mask should be 3D or 4D but is {ndim}D")

    original_shape = outliers_mask.shape

    # Convert to float for convolution
    mask = outliers_mask.float()

    if ndim == 3:  # 1D case
        # Create 1D structuring element
        se = torch.ones(mask.size(1), 1, window_size, device=mask.device)
        padded_mask = F.pad(mask, (0, window_size - 1), "constant", 0)
        # Apply convolution
        convolved = F.conv1d(padded_mask, se, padding=0, groups=mask.size(1))

    else:  # 2D case
        # Create 2D square structuring element
        se = torch.ones(mask.size(1), 1, window_size, window_size, device=mask.device)
        # Pad to avoid boundary effects
        padded_mask = F.pad(
            mask, (window_size - 1, 0, window_size - 1, 0), "constant", 0
        )
        # Apply convolution
        convolved = F.conv2d(padded_mask, se, padding=0, groups=mask.size(1))

    # Threshold to get binary mask
    result = (convolved > 0).bool()

    assert (
        result.shape == original_shape
    ), f"Shape mismatch: got {result.shape}, expected {original_shape}"

    return result


def get_outlier_mask(
    data,
    threshold,
    moving_average=None,
    opening_window=None,
    union_channels=True,
):
    if moving_average is not None:
        if not isinstance(moving_average, dict):
            moving_average = {}  # Apply with default parameters
        data = apply_moving_average(data, **moving_average)

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


def add_outliers_2d(X, contamination=0.1, patch_size=None, strength=0.8, seed=None):
    """
    Add outliers to 2D data.

    Parameters
    ----------
    X : torch.Tensor
        4D data of shape (n_trials, n_channels, height, width).
    patch_size : int, tuple, optional
        Size of the patch to add outliers, by default None.
        if int, the size of the patch is (patch_size, patch_size).
        if None, randomly select a patch size between 5 and 15% of the image size.
    seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    torch.Tensor
        Data with outliers
    torch.Tensor
        Mask indicating where outliers were added (1 for outliers, 0 for clean data)
    """
    # If numpy array, convert to tensor
    if not torch.is_tensor(X):
        X = torch.tensor(X)

    # If seed is not torch.int, convert to torch.int
    if seed is not None and not isinstance(seed, int):
        seed = int(seed)

    # Set up generator for reproducible randomness
    generator = None
    if seed is not None:
        generator = torch.Generator(device=X.device)
        generator.manual_seed(seed)

    n_trials, n_channels, height, width = X.shape

    X_outliers = X.clone()
    outlier_mask = torch.zeros_like(X, dtype=torch.int)
    running_contamination = 0
    ratio_contam = 0

    while ratio_contam < contamination:
        if patch_size is None:
            patch_size = (
                torch.randint(
                    int(0.05 * height), int(0.30 * height), (1,), generator=generator
                ).item(),
                torch.randint(
                    int(0.05 * width), int(0.30 * width), (1,), generator=generator
                ).item(),
            )

        start = (
            torch.randint(0, height - patch_size[0], (1,), generator=generator).item(),
            torch.randint(0, width - patch_size[1], (1,), generator=generator).item(),
        )
        end = start[0] + patch_size[0], start[1] + patch_size[1]

        # Using torch.rand instead of torch.rand_like with generator
        patch_shape = (n_trials, n_channels, end[0] - start[0], end[1] - start[1])
        random_patch = torch.rand(patch_shape, device=X.device, generator=generator)
        X_outliers[:, :, start[0] : end[0], start[1] : end[1]] += (
            strength * random_patch
        )

        outlier_mask[:, :, start[0] : end[0], start[1] : end[1]] = 1
        # Update the contamination ratio
        running_contamination += torch.prod(torch.tensor(patch_size))
        ratio_contam = running_contamination / (height * width)

    X_outliers = torch.clamp(X_outliers, 0, 1)

    return X_outliers, outlier_mask
