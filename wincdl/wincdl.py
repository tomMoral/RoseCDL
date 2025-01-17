import numpy as np
import torch
import warnings

from .loss import LassoLoss, OutlierLoss
from .model import CSC1d, CSC2d
from .optimizer import SLS
from .train import train

from .datasets import create_dataloader


class WinCDL(torch.nn.Module):
    """

    Parameters
    ----------
    n_components : int
        Number of atoms in the dictionary.
    kernel_size : int or tuple of int
        Size of the convolutional kernel. This is used to infer the dimensionality
        of the data.
    n_channels : int
        Number of channels in the data.
    lmbd : float
        Regularization parameter.
    scale_lmbd : bool, optional
        If True, the regularization parameter will be scaled by the maximum value
        it can take for the solution to be non-zero. This is useful to set the
        regularization parameter in the interval [0, 1].
    n_iterations : int, optional
        Number of iterations for the internal CSC algorithm.
    epochs : int, optional
        Number of epochs for the training.
    max_batch : int, optional
        Maximum number of batches to process per epoch. If None, all batches will
        be processed.
    optimizer : str, optional
        Name of the optimizer to use. Can be "adam" or "linesearch".
    lr : float, optional
        Learning rate for the optimizer.
    gamma : float, optional
        Learning rate decay for the Adam optimizer.
    sample_window : int or tuple of int, optional
        Size of minibatch windows. If int, the same window size will be used for
        each dimension. If tuple, the number of elements should match the data
        dimensionality. If None, no subwindows will be extracted and the dataset
        will return each trial as is.
    mini_batch_size : int, optional
        Size of the mini-batch.
    rank : str, optional
        Rank of the dictionary. Can be "full" or "rank1". "rank1" will only be
        used for 1D signals.
    window : bool, optional
        If True, the dictionary will be multiplied by a window function, to have
        its value on the border to be zero.
    D_init : array, optional
        Initial dictionary. If None, the dictionary will be initialized randomly.
    positive_z : bool, optional
        If True, the activations will be constrained to be positive.
    positive_D : bool, optional
        If True, the dictionary will be constrained to be positive.
        Useful for images.
    outliers_kwargs : dict, optional
        Parameters for the outliers detection. If None, no outliers detection will
        be used.
    callbacks : tuple, optional
        List of callbacks to use during the training. The callbacks should be callable
        functions with signature `callback(model, epoch, loss)`.
    device, dtype : str, optional
        Device and data type for the data. If None, the data will be converted to
        torch.tensor with default values.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components,
        kernel_size,
        n_channels,
        lmbd,
        scale_lmbd=True,
        n_iterations=30,
        epochs=100,
        max_batch=10,
        optimizer="linesearch",
        lr=0.1,
        gamma=0.9,
        sample_window=1000,
        mini_batch_size=10,
        rank="full",
        window=False,
        D_init=None,
        positive_z=True,
        positive_D=False,  # Add this parameter
        outliers_kwargs=None,
        callbacks=(),
        device=None,
        dtype=torch.float,
        random_state=2147483647,
    ):
        super().__init__()

        kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.dimN = len(kernel_size)

        self.lmbd = lmbd
        self.scale_lmbd = scale_lmbd

        self.epochs = epochs
        self.max_batch = max_batch
        self.mini_batch_size = mini_batch_size
        self.stochastic = sample_window is not None
        self.sample_window = sample_window
        self.optimizer_name = optimizer
        self.gamma = gamma
        self.callbacks = callbacks

        self.random_state = random_state
        self.dtype = dtype
        self.device = device

        self.loss_fn = LassoLoss(lmbd=lmbd, reduction="sum")
        if outliers_kwargs is not None:
            self.loss_fn = OutlierLoss(
                self.loss_fn,
                method=outliers_kwargs.get("method", "quantile"),
                alpha=outliers_kwargs.get("alpha", 0.05),
                moving_average=outliers_kwargs.get("moving_average", None),
                opening_window=outliers_kwargs.get("opening_window", True),
                union_channels=outliers_kwargs.get("union_channels", True),
            )

        # CSC solver
        csc_class = CSC1d if self.dimN == 1 else CSC2d

        self.csc = csc_class(
            n_iterations,
            n_components,
            kernel_size,
            n_channels,
            lmbd,
            rank=rank,
            window=window,
            D_init=D_init,
            positive_z=positive_z,
            positive_D=positive_D,
            random_state=self.random_state,
            device=device,
            dtype=dtype,
        )

        # Optimizer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr)
        elif optimizer == "linesearch":
            self.optimizer = SLS(self.parameters(), sto=self.stochastic, lr=lr)

        self.to(device=device)

    @property
    def D_hat_(self):
        return self.csc.D_hat_.copy()

    def check_X(self, X):

        if self.dimN == 1:
            # X should be of shape (batch, n_channels, support)
            expected_dim = 3
        elif self.dimN == 2:
            # X should be of shape (batch, n_channels, height, width)
            expected_dim = 4

        # If missing one dimension, add a channel dimension.
        # This will only work if n_channels=1
        if X.ndim == expected_dim - 1:
            X = X[:, None]

        assert X.ndim == expected_dim, (
            f"The input data X should be of dimension {expected_dim}, with shape "
            f"(batch_size, n_channels, *support). Got X of shape {X.shape}"
        )

        if X.shape[1] != self.csc.n_channels:
            raise ValueError(
                f"The number of channel in X do not match the one specified in "
                f"WinCDL. Got {X.shape[1]=}, while we expected {self.csc.n_channels}"
            )

        if any(supp < 3 * ks for supp, ks in zip(X.shape[2:], self.csc.kernel_size)):
            warnings.warn(
                "The support of the signal is smaller than 3 times the kernel size. "
                "This may lead to poor performance."
            )

        return X

    def fit(self, X):
        # Dataloader
        if isinstance(X, torch.utils.data.dataloader.DataLoader):
            train_dataloader = X  # quick fix to use on physionet
        else:
            # Generated Data
            X = self.check_X(X)  # Removes the channel dimension
            train_dataloader = create_dataloader(
                X,
                sample_window=self.sample_window,
                mini_batch_size=self.mini_batch_size,
                random_state=self.random_state,
                device=self.device,
                dtype=self.dtype,
            )

        if self.scale_lmbd:
            lambda_max = self.loss_fn.get_lambda_max(train_dataloader, self.csc.get_D())
            self.loss_fn.lmbd = self.lmbd * lambda_max
            self.csc.lmbd = self.lmbd * lambda_max

        print("Set up optimizer and scheduler...", end=" ")
        # LR scheduler
        if self.max_batch is None:
            self.max_batch = len(train_dataloader)

        if self.stochastic and self.optimizer_name == "adam":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, np.power(self.gamma, 1 / self.max_batch)
            )
        elif self.stochastic and self.optimizer_name == "linesearch":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs * self.max_batch
            )
        else:
            self.scheduler = None
        print("Done.")

        # Train
        train(
            self,
            train_dataloader,
            self.optimizer,
            self.loss_fn,
            scheduler=self.scheduler,
            epochs=self.epochs,
            max_batch=self.max_batch,
            callbacks=self.callbacks,
            stopping_criterion=not self.stochastic,
        )

        return self

    def predict(self, X):
        X = torch.tensor(X, device=self.device, dtype=self.dtype)
        X_hat, _ = self.csc(X)
        return X_hat.detach().cpu().numpy()
