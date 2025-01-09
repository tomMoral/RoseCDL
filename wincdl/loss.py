import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


from .utils.utils_outliers import get_outlier_mask, get_thresholds


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
        full - valid + 1 for full, valid in zip(X.shape[2:], z.shape[2:])
    )


def reduce_loss(loss, reduction):
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction != "none":
        raise ValueError(f"reduction={reduction} is not valid.")


class OutlierLoss(_Loss):
    def __init__(
        self, loss_fn, thresholds=None, moving_average=None, opening_window=True, union_channels=True,
        method="quantile", alpha=0.05, reduction=None
    ):
        """Trimmed loss with outliers detection.

        Parameters
        ----------
        loss_fn : torch.nn.Module
            The loss function to trimm to compute the loss.
        """
        super().__init__()

        self.loss_fn = loss_fn
        self.reduction = self.loss_fn.reduction if reduction is None else reduction
        self.loss_fn.reduction = 'none'

        self.thresholds = thresholds
        self.moving_average = moving_average
        self.opening_window = opening_window
        self.union_channels = union_channels
        self.device = loss_fn.device


    def forward(self, X_hat, z_hat, X, outliers_mask=None):
        if outliers_mask is None:
            with torch.no_grad():
                outliers_mask = self.get_outliers_mask(X_hat, z_hat, X)

        return reduce_loss(
            self.loss_fn(X_hat[~outliers_mask], X[~outliers_mask]),
            self.reduction
        )

    def get_outliers_mask(self, X_hat, z_hat, X):
        kernel_size = get_kernel_size(X_hat, z_hat)

        # Compute error vector, keep it 3D
        err = self.compute_patch_error(X_hat, z_hat, X)
        # Remove outliers
        return get_outlier_mask(
            data=err,
            thresholds=self.thresholds,
            moving_average=self.moving_average,
            opening_window=kernel_size if self.opening_window else None,
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
        patch_size : int
            The size of the patch to compute the error on.

        Returns
        -------
        err : torch.Tensor, shape (n_batch, *patch_support)
            The reconstruction error per patch.
        """
        kernel_size = get_kernel_size(X_hat, z_hat)

        # Compute non-reduced loss
        diff = self.loss_fn(X_hat, z_hat, X)

        avg_pool = F.avg_pool1d if X.ndim == 3 else F.avg_pool2d

        # Extend on right to get patched aligned with coefficient z
        pad_size = tuple(v for ks in kernel_size[::-1] for v in (0, ks-1))
        diff = F.pad(diff, pad_size, "constant", 0)
        diff = avg_pool(diff, kernel_size, stride=1)

        # Sum across channels
        diff = torch.sum(diff, dim=1)

        return diff

    def compute_outlier_thresholds(self, model, dataloader):
        with torch.no_grad():
            err_i_batches = []
            n_samples = 0
            for X in dataloader:
                X_hat, z_hat = model(X)
                err_i = self.compute_patch_error(X_hat, z_hat, X)
                err_i_batches.append(err_i.flatten())

                n_samples += err_i.size(0)
                if n_samples >= 300:
                    # Compute thresholds on minimum 100 samples
                    break

            err = torch.cat(err_i_batches)
            self.thresholds = get_thresholds(err, **self.outliers_kwargs)


class LassoLoss(_Loss):
    def __init__(self, lmbd, reduction='mean', data_fit=torch.nn.MSELoss()):
        super().__init__(reduction=reduction)
        self.data_fit = data_fit
        self.lmbd = lmbd

        self.data_fit.reduction = "none"

    def forward(self, X_hat, z_hat, X):
        kernel_size = get_kernel_size(X_hat, z_hat)

        # should be of shape (batch_size, *full_suppport)
        loss = self.data_fit(X_hat, X).sum(dim=1)

        # Compute the L1 norm and pad it to be able to add it to each patch
        if self.lmbd > 0:
            pad_size = tuple(v for ks in kernel_size[::-1] for v in (0, ks-1))
            z_hat = F.pad(z_hat.abs().sum(dim=1), pad_size, "constant", 0)
            loss += self.lmbd * z_hat

        loss = reduce_loss(loss, self.reduction)

        return loss
