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
        device=None,
        dtype=torch.float,
        random_state=2147483647,
        rank="full",
        window=False,
        D_init=None,
        positive_z=True,
        list_D=False,
        dimN=1,
        n_samples=None,
        outliers_kwargs=None,
        callbacks=(),
    ):
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
        self.dimN = dimN
        self.n_samples = n_samples
        self.callbacks = callbacks

        self.loss_fn = LassoLoss(lmbd=lmbd, reduction="sum")
        if outliers_kwargs is not None:
            self.loss_fn = OutlierLoss(
                self.loss_fn,
                moving_average=self.outliers_kwargs.get("moving_average", None),
                opening_window=self.outliers_kwargs.get("opening_window", True),
                union_channels=self.outliers_kwargs.get("union_channels", True),
            )

        # CSC solver
        csc_class = CSC1d if dimN == 1 else CSC2d

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
        # Check the dimensions of X and reshape it if necessary
        if X.ndim == 3:
            X = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
        elif X.ndim != 2:
            raise ValueError("X must be 2D or 3D.")

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