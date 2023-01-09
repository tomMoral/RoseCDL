import torch
import numpy as np

from .model import CSC1d, CSC2d
from .datasets import create_conv_dataloader
from .optimizer import SLS
from .train import train


class WinCDL:

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
        optimizer="adam",
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
        dimN=1
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

        # CSC solver
        if dimN == 1:
            self.csc = CSC1d(
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
                positive_z=positive_z
            )

        elif dimN == 2:
            self.csc = CSC2d(
                n_iterations,
                n_components,
                kernel_size,
                n_channels,
                lmbd,
                device,
                dtype,
                random_state=self.random_state,
                D_init=D_init,
                positive_z=positive_z,
            )

        # Optimizer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.csc.parameters(), lr)
        elif optimizer == "linesearch":
            self.optimizer = SLS(
                self.csc.parameters(),
                sto=stochastic,
                lr=lr
            )

    @property
    def D_hat_(self):
        return self.csc.D_hat_

    def fit(self, X):

        # Dataloader
        train_dataloader = create_conv_dataloader(
            X,
            self.device,
            self.dtype,
            sto=self.stochastic,
            window=self.mini_batch_window,
            mini_batch_size=self.mini_batch_size,
            random_state=self.random_state,
            dimN=self.dimN
        )

        # LR scheduler
        if self.max_batch is None:
            self.max_batch = len(train_dataloader)

        if self.stochastic and self.optimizer_name == "adam":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                np.power(self.gamma, 1 / self.max_batch)
            )
        elif self.stochastic and self.optimizer_name == "linesearch":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs * self.max_batch
            )
        else:
            self.scheduler = None

        # Train
        losses, list_D, times = train(
            self.csc,
            train_dataloader,
            self.optimizer,
            torch.nn.MSELoss(),
            scheduler=self.scheduler,
            epochs=self.epochs,
            max_batch=self.max_batch,
            save_list_D=self.list_D,
            stopping_criterion=not self.stochastic
        )

        return losses, list_D, times
