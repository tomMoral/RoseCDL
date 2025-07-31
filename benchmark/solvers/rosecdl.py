from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

import torch

from rosecdl.rosecdl import RoseCDL


class Solver(BaseSolver):
    name = "RoseCDL"

    parameters = {
        # sample_window is defined as a multiple of the atom_support
        "mini_batch_size": [128],
        "sample_window": [32],
        "n_csc_iterations": [50],
        "random_state": [None],
        "outliers_kwargs": [
            None,
            {"method": "mad", "alpha": 3.5},
        ],
        "optimizer": ["adam", "linesearch"],
    }

    stopping_criterion = SufficientProgressCriterion(patience=15, strategy="callback")

    def get_next(self, stop_val):
        return stop_val + 50

    def skip(self, X, D_init, reg, window, has_outliers):
        if not has_outliers and self.outliers_kwargs is not None:
            return True, "Don't run with outlier detection on data without outliers."

        return False, None

    def set_objective(
        self,
        X,
        D_init,
        reg,
        window,
        has_outliers,
    ):
        """Store the dataset information to use in `run`.

        Parameters
        ----------
        X : ndarray, (n_trials, n_channels, *support)
            The signals to encode with CDL.
        D_init : ndarray, (n_atoms, n_channels, *atom_support)
            The initial dictionary, specified for the problem.
        reg : float
            The regularization parameter for the problem. This parameter will be
            scaled by its maximum value for the data, so it should be in the range
            [0, 1]. **Note**: we use this convention so this can be adapted for
            methods which use annomaly detection.
        window : bool
            Whether to use a windowed dictionary or not.
        has_outliers : bool
            Whether the data has outliers or not.

        """
        self.X, self.D_init, self.reg = X, D_init, reg
        self.window = window
        self.has_outliers = has_outliers

        if not isinstance(self.X, torch.Tensor):
            self.X = torch.tensor(self.X, dtype=torch.float32)
        if not isinstance(self.D_init, torch.Tensor):
            self.D_init = torch.tensor(self.D_init, dtype=torch.float32)

        # Infer dictionary size from D_init
        self.n_atoms = D_init.shape[0]
        self.n_channels = self.X.shape[1]
        self.atom_support = D_init.shape[2:]

        sample_window = tuple(self.sample_window * s for s in self.atom_support)
        rank1 = D_init.ndim == 2

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_kwargs = dict(
            lmbd=self.reg,
            scale_lmbd=False,
            epochs=10000,
            max_batch=None,
            mini_batch_size=self.mini_batch_size,
            sample_window=sample_window,
            D_init=self.D_init,
            window=self.window,
            rank1=rank1,
            outliers_kwargs=self.outliers_kwargs,
            optimizer=self.optimizer,
            n_iterations=self.n_csc_iterations,
            random_state=self.random_state,
            device=self.device,
        )

    def run(self, cb):
        self.model = RoseCDL(**self.model_kwargs, callbacks=[lambda *x: not cb()])
        cb()  # Get init value
        self.model.fit(self.X)

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all operations are done before returning
        return {"D": self.model.D_hat_}
