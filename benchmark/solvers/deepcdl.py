from benchopt import BaseSolver

from benchopt.stopping_criterion import SufficientProgressCriterion

import torch
from rosecdl.rosecdl import RoseCDL


class Solver(BaseSolver):

    name = "DeepCDL"

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
        return stop_val + 3

    def set_objective(self, X, D_init, reg, window, has_outliers):

        self.X, self.reg, self.D_init = X, reg, D_init
        self.window = window
        self.has_outliers = has_outliers

        # Infer dictionary size from D_init
        self.n_atoms = D_init.shape[0]
        self.n_channels = self.X.shape[1]
        self.atom_support = D_init.shape[2:]

        # Scale sample window with the size of the atom
        sample_window = tuple(self.sample_window * s for s in self.atom_support)

        rank1 = D_init.ndim == 2

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_kwargs = dict(
            lmbd=self.reg,
            scale_lmbd=False,
            D_init=self.D_init,
            window=self.window,
            rank1=rank1,
            outliers_kwargs=None,
            epochs=10000,
            max_batch=None,
            mini_batch_size=self.mini_batch_size,
            sample_window=sample_window,
            deepcdl=True,
            optimizer=self.optimizer,
            n_iterations=self.n_csc_iterations,
            random_state=self.random_state,
            device=self.device,
        )

    def run(self, cb):
        X = self.X
        if self.outliers_kwargs is not None:
            X = remove_outliers_before_cdl(X, self.z_shape, **self.outliers_kwargs)
        self.model = RoseCDL(**self.model_kwargs, callbacks=[lambda *x: not cb()])
        cb()  # Get init value
        self.model.fit(X)

    def get_result(self):
        return {"D": self.model.D_hat_}
