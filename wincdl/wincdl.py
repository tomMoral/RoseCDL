import numpy as np
import torch

from .datasets import create_conv_dataloader
from .model import CSC1d, CSC2d
from .optimizer import SLS
from .train import train
from .loss import OutlierLoss, LassoLoss


class WinCDL:
    """

    uv_constraint

    """

    def __init__(
        self,
        n_components,
        kernel_size,
        n_channels,
        lmbd,
        n_iterations=30,
        epochs=100,
        max_batch=10,
        stochastic=False,
        optimizer="linesearch",
        lr=0.1,
        gamma=0.9,
        mini_batch_window=1000,
        mini_batch_size=10,
        random_state=2147483647,
        rank="full",
        window=False,
        D_init=None,
        positive_z=True,
        list_D=False,
        n_samples=None,
        outliers_kwargs=None,
        callbacks=(),
        device=None,
        dtype=torch.float,
    ):

        kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size

        self.stochastic = stochastic
        self.mini_batch_window = mini_batch_window
        self.mini_batch_size = mini_batch_size
        self.random_state = random_state
        self.epochs = epochs
        self.max_batch = max_batch
        self.device = device
        self.dtype = dtype
        self.list_D = list_D
        self.gamma = gamma
        self.optimizer_name = optimizer
        self.dimN = len(kernel_size)
        self.n_samples = n_samples
        self.callbacks = callbacks

        self.loss_fn = LassoLoss(lmbd=lmbd, reduction="sum")
        if outliers_kwargs is not None:
            self.loss_fn = OutlierLoss(
                self.loss_fn,
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
            device,
            dtype,
            random_state=self.random_state,
            rank=rank,
            window=window,
            D_init=D_init,
            positive_z=positive_z,
        )

        # Optimizer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.csc.parameters(), lr)
        elif optimizer == "linesearch":
            self.optimizer = SLS(self.csc.parameters(), sto=stochastic, lr=lr)

    @property
    def D_hat_(self):
        return self.csc.D_hat_.copy()

    def check_X(self, X):
        # TODO: check this is always of shape (batch, n_channels, *support)
        # Check the dimensions of X and reshape it if necessary
        if X.ndim == 3:
            X = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
        elif X.ndim != 2:
            raise ValueError("X must be 2D or 3D.")

        return X

    def get_lambda_max(self, X):
        """For each atom, compute the regularization parameter scaling.
        This value is usually defined as the smallest value for which 0 is
        a solution of the optimization problem.
        In order to avoid spurious values, this quantity can also be estimated
        as the q-quantile of the correlation between signal patches and the
        atom.

        Parameters
        ----------
        X : array, shape (n_trials, n_times) or
                shape (n_trials, n_channels, n_times)
            The data

        alpha : float
            If method is quantile (default), the quantile to compute, which must be between 0 and 1 inclusive.
            Default is 1, i.e., the maximum is returned.
            If method is iqr, zscore or mad, the alpha parameter to use.

        Returns
        -------
        lambda_max : array, shape (n_atoms, 1)
        """
        conv_res = self.csc.conv(X, self.csc.D_hat_).abs().cpu().numpy()

        # TODO: this should reuse get_threshold
        if self.loss_fn.method == "quantile":
            assert self.loss_fn.alpha <= 1 and self.loss_fn.alpha >= 0
            return np.quantile(conv_res, axis=(1, 2), q=self.loss_fn.alpha)[:, None]
        elif self.loss_fn.method == "iqr":
            assert self.loss_fn.alpha >= 1
            q1, q3 = np.quantile(conv_res, axis=(1, 2), q=[0.25, 0.75])
            res = q3 + 1.5 * (q3 - q1)
            return res[:, None]
        elif self.loss_fn.method == "zscore":
            assert self.loss_fn.alpha >= 1
            res = np.mean(conv_res, axis=(1, 2)) + self.loss_fn.alpha * np.std(conv_res, axis=(1, 2))
            return res[:, None]
        elif self.loss_fn.method == "mad":
            assert self.loss_fn.alpha >= 1
            median = np.median(conv_res, axis=(1, 2))
            mad = np.median(np.abs(conv_res - median[:, None, None]), axis=(1, 2))
            constant = 0.6745
            res = median + self.loss_fn.alpha * mad / constant
            return res[:, None]

    def fit(self, X):
        # Dataloader
        if isinstance(X, torch.utils.data.dataloader.DataLoader):
            train_dataloader = X  # quick fix to use on physionet
        else:
            # Generated Data
            X = self.check_X(X)  # Removes the channel dimension
            train_dataloader = create_conv_dataloader(
                X,
                self.device,
                self.dtype,
                sto=self.stochastic,
                window=self.mini_batch_window,
                mini_batch_size=self.mini_batch_size,
                random_state=self.random_state,
                dimN=self.dimN,
                n_samples=self.n_samples,
            )

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
        X_hat, _ = self.csc(X)
        return X_hat.detach().cpu().numpy()