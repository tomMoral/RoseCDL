import numpy as np
import torch
import torch.nn.functional as F

from rosecdl.utils.utils import get_torch_generator


def check_threshold(threshold: float | torch.Tensor | None) -> None:
    """Validate the type and value of a threshold parameter.

    This function checks if threshold is either None, a float/int, or PyTorch tensor
    with appropriate numeric datatypes.

    Args:
        threshold: The threshold to validate. None, a float/int, or a PyTorch tensor
            with dtype float32 or int32.

    Raises:
        ValueError: If the threshold is not None and is neither a float/int nor a
            PyTorch tensor with appropriate dtype.

    Returns:
        None

    """
    # Ensure that the threshold is either float or None
    if threshold is not None:
        if isinstance(threshold, torch.Tensor):
            if threshold.dtype not in (torch.float32, torch.int32):
                msg = (
                    f"threshold.dtype must be float32 or int32 but got "
                    f"{threshold.dtype}"
                )
                raise ValueError(msg)
        elif not isinstance(threshold, float | int):
            msg = f"threshold should be a float or None but is {type(threshold)}"
            raise ValueError(msg)


def get_threshold(
    data: torch.Tensor | np.ndarray, method: str = "quantile", alpha: float = 0.05
) -> float:
    """Compute the outlier detection threshold.

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
    - 'quantile': Upper threshold are determined by the specified quantile levels.
    - 'iqr': Threshold are determined based on the whiskers of the interquartile range.
    - 'zscore': The threshold are determined based on the mean and standard deviation.
                A value is considered an outlier if it's alpha standard deviations
                away from the mean.
    - 'mad': The threshold are determined based on the median and median absolute
             deviation. A value is considered an outlier if it's alpha median absolute
             deviations away from the median.

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
        # "The constant 0.6745 is needed because E(MAD) = 0.6745CT for large n",  # noqa: ERA001, E501
        # Iglewicz and Hoaglin, 1993
        # Calculate lower and upper threshold
        threshold = median + alpha * mad / constant

    else:
        # Raise an error if an unsupported method is chosen
        msg = (
            f"Invalid method: {method}, must be one of 'quantile', "
            "'iqr', 'zscore', or 'mad'"
        )
        raise ValueError(msg)

    threshold = threshold.item()
    check_threshold(threshold)

    return threshold


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Generate a 1D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
    return gauss / gauss.sum()


def gaussian_kernel_2d(size: int, sigma: float) -> torch.Tensor:
    """Generate a 2D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    y = x.unsqueeze(0)
    x = x.unsqueeze(1)
    gauss = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma**2))
    return gauss / gauss.sum()


def apply_moving_average(
    data: torch.Tensor, window_size: int = 15, method: str = "average"
) -> torch.Tensor:
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
        msg = f"Data must be 3D or 4D but got {ndim}D"
        raise ValueError(msg)

    if method not in ["average", "gaussian", "max"]:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

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

    elif method == "average":
        se = torch.ones(
            data.size(1), 1, window_size, window_size, device=data.device
        ) / (window_size * window_size)
        result = F.conv2d(data, se, padding=window_size // 2, groups=data.size(1))

    elif method == "gaussian":
        sigma = window_size / 3
        kernel = gaussian_kernel_2d(window_size, sigma).to(data.device)
        se = kernel.view(1, 1, window_size, window_size).repeat(data.size(1), 1, 1, 1)
        result = F.conv2d(data, se, padding=window_size // 2, groups=data.size(1))

    else:  # method == "max"
        result = F.max_pool2d(data, window_size, stride=1, padding=window_size // 2)

    if result.shape != original_shape:
        msg = f"Shape mismatch: got {result.shape}, expected {original_shape}"
        raise ValueError(msg)

    return result


def apply_opening(outliers_mask: torch.Tensor, window_size: int = 15) -> torch.Tensor:
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
    if not isinstance(window_size, int | tuple) or window_size <= 0:
        return outliers_mask

    ndim = outliers_mask.ndim
    if ndim not in [3, 4]:
        msg = f"outliers_mask should be 3D or 4D but is {ndim}D"
        raise ValueError(msg)

    original_shape = outliers_mask.shape

    # Convert to float for convolution
    mask = outliers_mask.float()

    conv = F.conv1d if ndim == 3 else F.conv2d

    if isinstance(window_size, int):
        if ndim == 3:
            window_size = (window_size,)
        else:
            window_size = (window_size, window_size)

    se = torch.ones(mask.size(1), 1, *window_size, device=mask.device)
    pad_size = tuple(v for ks in window_size[::-1] for v in (ks // 2, (ks - 1) // 2))
    padded_mask = F.pad(mask, pad_size, "constant", 0)
    convolved = conv(padded_mask, se, padding=0, groups=mask.size(1))

    # Threshold to get binary mask
    result = (convolved > 0).bool()

    if result.shape != original_shape:
        msg = f"Shape mismatch: got {result.shape}, expected {original_shape}"
        raise ValueError(msg)

    return result


def get_outlier_mask(
    data: torch.Tensor,
    threshold: float | torch.Tensor | None,
    moving_average: dict | None = None,
    opening_window: int | None = None,
    union_channels: bool = True,
) -> torch.Tensor:
    """Generate a mask identifying outlier values in tensor data.

    This function identifies values in the data that exceed a specified threshold,
    with options for preprocessing using moving averages and post-processing
    using morphological operations.

    Parameters
    ----------
    data : torch.Tensor
        The input data tensor to check for outliers.
    threshold : float or torch.Tensor or None
        The threshold value to identify outliers. Values in data exceeding
        this threshold are considered outliers.
    moving_average : dict or None, optional
        Parameters for applying a moving average filter to the data before
        outlier detection. If None, no moving average is applied. Default is None.
    opening_window : int or None, optional
        Window size for applying morphological opening to the outlier mask.
        If None, no opening operation is performed. Default is None.
    union_channels : bool, optional
        If True and input has 3 dimensions, takes the boolean union across channels.
        Default is True.

    Returns
    -------
    torch.Tensor
        A boolean mask tensor with the same shape as input data, where
        True values indicate outliers.

    Raises
    ------
    ValueError
        If the shape of the resulting outlier mask doesn't match the input data shape.

    """
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

    if outliers_mask.shape != data.shape:
        msg = (
            f"Shape mismatch: outliers_mask.shape: {outliers_mask.shape}, "
            f"data.shape: {data.shape}"
        )
        raise ValueError(msg)

    return outliers_mask


def remove_outliers(
    data: torch.Tensor,
    threshold: float | torch.Tensor | None = None,
    method: str = "quantile",
    opening_window: int | None = None,
    moving_average: dict | None = None,
    n_channels: int = 1,
    **kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove outliers from a batch vector.

    Parameters
    ----------
    data : torch.Tensor
        Input data tensor from which to remove outliers.
    threshold : float | torch.Tensor | None, optional
        Threshold value for outlier detection. If None, computed based on method.
    method : str, optional
        Method for outlier detection, by default "quantile".
        Options: 'quantile', 'iqr', 'zscore', or 'mad'.
    opening_window : int | None, optional
        Size of the opening window to remove isolated outliers, by default None.
    moving_average : dict, optional
        Moving average parameters, by default None
        example: moving_average=dict(
            window_size=int(model.n_times_atom),
            method='max',  # 'max' or 'average'
            gaussian=False,
        )
    n_channels : int, optional
        Number of channels in the data, by default 1.
    **kwargs : dict
        Additional arguments to pass to underlying functions.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing:
        - Cleaned data with outliers removed
        - Boolean mask indicating where outliers were detected

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


def add_outliers_2d(
    X: torch.Tensor,
    contamination: float = 0.1,
    patch_size: tuple[int, int] | int | None = None,
    strength: float = 0.8,
    seed: int | None = None,
    noise: float | None = None,
    clip: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add outliers to 2D data.

    Parameters
    ----------
    X : torch.Tensor
        4D data of shape (n_trials, n_channels, height, width).
    contamination : float, optional
        Proportion of outliers to add, by default 0.1
    patch_size : int, tuple, optional
        Size of the patch to add outliers, by default None.
        if int, the size of the patch is (patch_size, patch_size).
        if None, randomly select a patch size between 5 and 15% of the image size.
    strength : float, optional
        Strength of the outliers, by default 0.8
    seed : int, optional
        Random seed for reproducibility, by default None
    clip : bool, optional
        Clip the data to [0, 1] after adding outliers, by default True

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
    generator = get_torch_generator(seed)

    n_trials, n_channels, height, width = X.shape

    X_outliers = X.clone()
    outlier_mask = torch.zeros_like(X, dtype=torch.int)
    running_contamination = 0
    ratio_contam = 0

    if noise is not None:
        # Not using torch.rand_like because of generator
        X_outliers += noise * torch.randn(
            X_outliers.shape, device=X.device, generator=generator
        )

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

    if clip:
        X_outliers = torch.clamp(X_outliers, 0, 1)

    return X_outliers, outlier_mask
