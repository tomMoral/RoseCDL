import numpy as np
import torch

from .datasets import create_conv_dataloader
from .loss import LassoLoss, OutlierLoss
from .model import CSC1d, CSC2d
from .optimizer import SLS
from .train import train


class WinCDL(torch.nn.Module):
    """

    uv_constraint

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
        n_samples=None,
        outliers_kwargs=None,
        callbacks=(),
        device=None,
        dtype=torch.float,
    ):
        super().__init__()

        kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.dimN = len(kernel_size)

        self.lmbd = lmbd
        self.scale_lmbd = scale_lmbd

        self.stochastic = stochastic
        self.epochs = epochs
        self.max_batch = max_batch
        self.mini_batch_size = mini_batch_size
        self.mini_batch_window = mini_batch_window
        self.gamma = gamma
        self.optimizer_name = optimizer
        self.n_samples = n_samples
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
            random_state=self.random_state,
            device=device,
            dtype=dtype,
        )

        # Optimizer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr)
        elif optimizer == "linesearch":
            self.optimizer = SLS(self.parameters(), sto=stochastic, lr=lr)

        self.to(device=device)

    @property
    def D_hat_(self):
        return self.csc.D_hat_.copy()

    def check_X(self, X):
        # Check this is always of shape (batch, n_channels, *support)
        if self.dimN == 1:
            # X should be of shape (batch, n_channels, support)
            if X.ndim == 2:
                X = X[:, None, :]
        elif self.dimN == 2:
            # X should be of shape (batch, n_channels, height, width)
            if X.ndim == 3:
                X = X[:, None, :, :]
        expected_shape = (X.shape[0], self.csc.n_channels, *self.csc.kernel_size)
        if X.shape != expected_shape:
            raise ValueError(f"Expected X shape {expected_shape}, but got {X.shape}")

        return X

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
