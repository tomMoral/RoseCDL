import numpy as np
import torch
from tqdm import tqdm

from .datasets import create_conv_dataloader
from .model import CSC1d, CSC2d
from .optimizer import SLS
from .train import compute_loss, train


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
        self.outliers_kwargs = outliers_kwargs

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
                positive_z=positive_z,
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
            self.optimizer = SLS(self.csc.parameters(), sto=stochastic, lr=lr)

    @property
    def D_hat_(self):
        return self.csc.D_hat_

    def check_X(self, X):
        # Check the dimensions of X and reshape it if necessary
        if X.ndim == 3:
            old_shape = X.shape
            X = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
            # warnings.warn(
            #     f"X shape was {old_shape}, reshaped it, now of shape {X.shape}"
            # )
        elif X.ndim != 2:
            raise ValueError("X must be 2D or 3D.")

        return X

    def fit(self, X):
        X = self.check_X(X)

        # Dataloader
        if isinstance(X, torch.utils.data.dataloader.DataLoader):
            train_dataloader = X  # quick fix to use on physionet
        else:
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
        try:
            self.subjects = train_dataloader.dataset.subjects
        except:
            pass

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
        train_hist = train(
            self.csc,
            train_dataloader,
            self.optimizer,
            torch.nn.MSELoss(reduction="sum"),
            scheduler=self.scheduler,
            epochs=self.epochs,
            max_batch=self.max_batch,
            save_list_D=self.list_D,
            stopping_criterion=not self.stochastic,
            outliers_kwargs=self.outliers_kwargs,
        )

        losses = train_hist["train_losses"]
        list_D = train_hist["list_D"]
        times = train_hist["times"]

        self.train_hist = train_hist

        return losses, list_D, times

    def compute_loss_hist(self, X, list_D=None, adapt_alphacsc=False):
        current_D = self.csc.D_hat_
        assert X.ndim == 3, "X must be 3D, of shape (n_trials, n_channels, n_times)."

        # Ensure X is a torch tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)

        if list_D is None:
            list_D = self.train_hist["list_D"]

        losses = []
        pbar = tqdm(list_D)
        pbar.set_description(f"Running loss computation for {len(list_D)} Ds.")
        for D in pbar:
            self.csc.set_D(D)
            loss = compute_loss(
                self.csc, self.loss_fn, X, adapt_alphacsc=adapt_alphacsc
            )
            losses.append(loss.item())

        self.csc.set_D(current_D)
        return losses

    def get_prediction(self, X, D=None):
        # Ensure X and D are torch tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)

        if D is not None and not isinstance(D, torch.Tensor):
            D = torch.tensor(D, dtype=self.dtype, device=self.device)

        X_hat = self.csc.forward(X, D)
        z = self.csc._z_hat

        old_reg = self.csc.lmbd
        self.csc.lmbd = 0

        threshold = 0
        null_support = torch.where(torch.abs(z) <= threshold)
        X_hat = self.csc.forward(X, D, null_support=null_support)

        self.csc.lmbd = old_reg

        return X_hat.detach().cpu().numpy()

    def compute_z_hat_history(self, X, list_D=None):
        # Ensure X and D are torch tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)

        old_reg = self.csc.lmbd
        self.csc.lmbd = 0

        z_hat_history = []

        if list_D is None:
            list_D = self.train_hist["list_D"]

        pbar = tqdm(list_D)
        pbar.set_description(f"Running z_hat computation for {len(list_D)} Ds.")
        for D in pbar:
            if D is not None and not isinstance(D, torch.Tensor):
                D = torch.tensor(D, dtype=self.dtype, device=self.device)

            _ = self.csc.forward(X, D)
            z_hat_history.append(self.csc.z_hat_)

        self.csc.lmbd = old_reg

        return z_hat_history
