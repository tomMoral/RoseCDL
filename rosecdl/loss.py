import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from rosecdl.utils.utils_outliers import get_outlier_mask, get_threshold


def get_kernel_size(X, z):
    """Get the kernel's size from the signal and sparse code.

    Paramters
    ---------
    X : torch.Tensor, shape (batch_size, n_channels, *full_support)
        The signal tensor.
    z : torch.Tensor, shape (batch_size, n_atoms, *valid_support)
        The sparse code tensor.

    Returns
    -------
    kernel_size : tuple
        The kernel's size corresponding for each dimension to
        full_support - valid_support + 1.

    """
    return tuple(
        full - valid + 1 for full, valid in zip(X.shape[2:], z.shape[2:], strict=False)
    )


def reduce_loss(loss, reduction):
    """Reduce the loss according to the reduction parameter.

    Parameters
    ----------
    loss : torch.Tensor
        The loss tensor to reduce.
    reduction : str
        The reduction method. Can be "none", "mean", "sum".

    """
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction != "none":
        raise ValueError(f"reduction={reduction} is not valid.")
    return loss


class _ReconstructionLoss(_Loss):
    def get_lambda_max(self, dataloader, D):
        """Compute the maximum value of the regularization parameter.

        This value is defined as the minimum value of the regularization parameter
        for which the solution is the zero vector.

        Parameters
        ----------
        dataloader : array, shape (batch_size, n_channels, *support)
            The signal tensor.
        D : array, shape (n_atoms, n_channels, *kernel_size)
            The dictionary tensor.

        Returns
        -------
        lambda_max : float
            The maximum value of the regularization parameter.

        """
        conv = F.conv1d if D.ndim == 3 else F.conv2d
        with torch.no_grad():
            n_samples, conv_res_i_batches = 0, []
            for X in dataloader:
                conv_res_i_batches.append(conv(X, D).abs().flatten())
                n_samples += X.size(0)
                if n_samples >= 300:
                    # Compute threshold on maximum 300 samples
                    break

            conv_res = torch.cat(conv_res_i_batches)
            if hasattr(self, "method"):
                lmbd_max = get_threshold(conv_res, self.method, self.alpha)
            else:
                lmbd_max = conv_res.max()
        return lmbd_max


class OutlierLoss(_ReconstructionLoss):
    """Base class for outlier detection loss."""

    def __init__(
        self,
        loss_fn,
        method="quantile",
        alpha=0.05,
        reduction=None,
        moving_average=None,
        opening_window=True,
        union_channels=True,
    ):
        """Trimmed loss with outliers detection.

        Parameters
        ----------
        loss_fn : torch.nn.Module
            The loss function to trimm to compute the loss.
        method : str, default="quantile"
            Outlier detection method. One of:
            - "quantile": Use quantile-based thresholding
            - "iqr": Use interquartile range
            - "zscore": Use z-score thresholding
            - "mad": Use median absolute deviation
        alpha : float, default=0.05
            Outlier detection parameter.
        reduction : str, optional
            Reduction method to apply ("mean", "sum", or "none").
            Defaults to the reduction method of loss_fn.
        moving_average : dict, optional
            Parameters for moving average smoothing:
            - window_size: Size of averaging window
            - method: Averaging method ("mean" or "max")
            - gaussian: Whether to use Gaussian weighting
        opening_window : bool, default=True
            Whether to apply morphological opening when calculating thresholds
        union_channels : bool, default=True
            Whether to detect outliers jointly across channels

        Attributes
        ----------
        _threshold : float
            Current outlier detection threshold (None until computed)

        """
        super().__init__()

        self.loss_fn = loss_fn
        self.reduction = self.loss_fn.reduction if reduction is None else reduction
        self.loss_fn.reduction = "none"

        self.method = method
        self.alpha = alpha

        self.moving_average = moving_average
        self.opening_window = opening_window
        self.union_channels = union_channels

        self._threshold = None

    def forward(self, X_hat, z_hat, X, outliers_mask=None):
        if outliers_mask is None:
            with torch.no_grad():
                outliers_mask = self.get_outliers_mask(X_hat, z_hat, X)

        return reduce_loss(
            self.loss_fn(X_hat, z_hat, X)[~outliers_mask], self.reduction
        )

    def get_outliers_mask(self, X_hat, z_hat, X, opening=None):
        kernel_size = get_kernel_size(X_hat, z_hat)
        opening = self.opening_window if opening is None else opening

        # Compute error vector, keep it 3D
        err = self.compute_patch_error(X_hat, z_hat, X)

        # Make sure we have the threshold
        threshold = self._threshold
        if threshold is None:
            threshold = get_threshold(err, self.method, self.alpha)

        # Remove outliers
        return get_outlier_mask(
            data=err,
            threshold=threshold,
            moving_average=self.moving_average,
            opening_window=kernel_size if opening else None,
            union_channels=self.union_channels,
        )

    def compute_patch_error(self, X_hat, z_hat, X):
        """Compute the reconstruction error per patch.

        Parameters
        ----------
        X_hat : torch.Tensor, shape (n_batch, n_channels, *full_support)
            The reconstructed signal tensor.
        z_hat : torch.Tensor, shape (n_batch, n_atoms, *valid_support)
            The sparse code tensor.
        X : torch.Tensor, shape (n_batch, n_channels, *full_support)
            The original signal tensor.

        Returns
        -------
        err : torch.Tensor, shape (n_batch, n_channels, *patch_support)
            The reconstruction error per patch, maintaining channel dimension.

        """
        kernel_size = get_kernel_size(X_hat, z_hat)

        # Compute non-reduced loss
        old_red = self.loss_fn.reduction
        self.loss_fn.reduction = "none"
        diff = self.loss_fn(X_hat, z_hat, X)
        self.loss_fn.reduction = old_red

        avg_pool = F.avg_pool1d if X.ndim == 3 else F.avg_pool2d

        # Extend on both sides to get patches aligned with coefficient z
        pad_size = tuple(
            v for ks in kernel_size[::-1] for v in (ks // 2, (ks - 1) // 2)
        )
        diff = F.pad(diff, pad_size, "constant", 0)
        return avg_pool(diff, kernel_size, stride=1)

    def compute_outlier_threshold(self, model, dataloader):
        with torch.no_grad():
            err_i_batches = []
            n_samples = 0
            for X in dataloader:
                X_hat, z_hat = model(X)
                err_i = self.compute_patch_error(X_hat, z_hat, X)
                err_i_batches.append(err_i.flatten())

                n_samples += err_i.size(0)
                if n_samples >= 300:
                    # Compute threshold on maximum 300 samples
                    break

            err = torch.cat(err_i_batches)
            self._threshold = get_threshold(err, method=self.method, alpha=self.alpha)


class LassoLoss(_ReconstructionLoss):
    """Lasso loss."""

    def __init__(self, lmbd, reduction="mean", data_fit=None):
        """Initialize the object.

        Args:
            lmbd: regularization parameter.
            reduction: reduction method. options are the same as for torch losses.
            data_fit (torch loss): default = MSE loss.

        """
        if data_fit is None:
            data_fit = torch.nn.MSELoss()
        super().__init__(reduction=reduction)
        self.data_fit = data_fit
        self.lmbd = lmbd

        self.data_fit.reduction = "none"

    def forward(self, X_hat, z_hat, X):
        kernel_size = get_kernel_size(X_hat, z_hat)

        # should be of shape (batch_size, 1, *full_suppport)
        loss = self.data_fit(X_hat, X).sum(dim=1, keepdim=True)

        # Compute the L1 norm and pad it to be able to add it to each patch
        if self.lmbd > 0:
            pad_size = tuple(v for ks in kernel_size[::-1] for v in (0, ks - 1))
            z_hat = F.pad(z_hat.abs().sum(dim=1, keepdim=True), pad_size, "constant", 0)
            loss += self.lmbd * z_hat

        return reduce_loss(loss, self.reduction)
