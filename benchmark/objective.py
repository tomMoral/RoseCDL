from pathlib import Path

from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from alphacsc.loss_and_gradient import compute_objective
    from alphacsc.update_z_multi import update_z_multi
    from alphacsc.utils.convolution import construct_X_multi
    from alphacsc.utils.dictionary import get_lambda_max

    from rosecdl.utils.utils_exp import evaluate_D_hat


ROSECDL_DIR = Path(__file__).parent.parent


class Objective(BaseObjective):
    name = "RoseCDL"

    # URL of the main repo for this benchmark.
    url = "https://github.com/tommoral/RoseCDL"

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.6.1"

    install_cmd = "conda"
    requirements = ["pip::alphacsc", f"pip::-e {ROSECDL_DIR}"]

    parameters = {
        "reg": [1e-1, 3e-1, 8e-1],
    }

    def set_data(self, X, D, D_init=True, window=True, outliers=None):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.D = D
        n_samples = len(X)
        if len(X) == 1:
            T_train = int(X.shape[2] * 0.8)
            self.X = X[:, :, :T_train]
            self.X_val = X[:, :, T_train:]
        else:
            n_train = max(int(n_samples * 0.8), 1)
            self.X = X[:n_train]
            self.X_val = X[n_train:]

        self.D_init = D_init.copy()
        self.outliers = outliers
        self.window = window
        self.scaled_reg = self.reg * get_lambda_max(X, D_init).max()

        # Auxillary variable to warm start the optimization for the loss computation
        self.z0_dict_ = {"train": None, "val": None}

    def _compute_objective(self, D, X, z0):
        z_hat, _, _ = update_z_multi(
            X,
            D.astype(float),
            reg=self.scaled_reg,
            z0=z0,
            solver="lgcd",
            solver_kwargs={"tol": 1e-2},
            n_jobs=min(8, X.shape[0]),
        )
        X_hat = construct_X_multi(z_hat, D=D)

        return (
            compute_objective(X, X_hat=X_hat, z_hat=z_hat, D=D, reg=self.scaled_reg),
            z_hat,
        )

    def evaluate_result(self, D):
        loss, z_train = self._compute_objective(D, self.X, self.z0_dict_["train"])
        loss_val, z_val = self._compute_objective(D, self.X_val, self.z0_dict_["val"])

        # Update warmstart
        self.z0_dict_.update(train=z_train, val=z_val)

        # Default value is the validation loss
        res = dict(loss=loss, loss_val=loss_val, value=loss_val)

        # Evaluate recovery of the dictionary
        if self.D is not None:
            recovery_score = evaluate_D_hat(self.D, D)
            res.update(recovery_score=recovery_score, value=-recovery_score)

        if isinstance(self.outliers, np.ndarray):
            # TODO: Add outlier detection
            pass

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return res

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(D=self.D_init)

    def get_objective(self):
        return dict(
            X=self.X,
            D_init=self.D_init,
            reg=self.scaled_reg,
            window=self.window,
            has_outliers=self.outliers is not None,
        )
