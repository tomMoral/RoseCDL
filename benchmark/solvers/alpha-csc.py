from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from alphacsc.convolutional_dictionary_learning import BatchCDL, GreedyCDL
    from alphacsc.online_dictionary_learning import OnlineCDL

    from rosecdl.utils.utils_outlier_comparison import remove_outliers_before_cdl

    ALGORITHMS = {
        "online": OnlineCDL,
        "batch": BatchCDL,
        "greedy": GreedyCDL,
    }


class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "alphaCSC"

    install_cmd = "conda"
    requirements = ["pip:alphacsc"]

    parameters = {
        "type": ["batch", "online"],
        "outliers_kwargs": [
            None,
            {"method": "quantile", "alpha": 0.2},
            {"method": "iqr", "alpha": 1.5},
            {"method": "mad", "alpha": 3.5},
            {"method": "zscore", "alpha": 1.5},
        ],
    }
    stopping_criterion = SufficientProgressCriterion(patience=10, strategy="callback")

    def get_next(self, stop_val):
        return stop_val + 1

    def skip(self, X, D_init, reg, window, has_outliers):
        if not has_outliers and self.outliers_kwargs is not None:
            return True, "Don't run with outlier detection on data without outliers."

        return False, None

    def set_objective(self, X, D_init, reg, window, has_outliers):
        self.X = X
        self.D_init = D_init
        self.reg = reg
        self.window = window
        self.has_outliers = has_outliers

        self.z_shape = tuple(
            xs - ds + 1 for xs, ds in zip(X.shape[2:], D_init.shape[2:])
        )
        self.z_shape = (X.shape[0], D_init.shape[0], *self.z_shape)

        # Rank1 dictionary if it is 2D
        rank1 = D_init.ndim == 2

        self.cdl = ALGORITHMS[self.type](
            n_atoms=D_init.shape[0],
            n_times_atom=D_init.shape[2],
            D_init=D_init.copy(),
            reg=reg,
            lmbd_max="scaled",
            solver_z="lgcd",
            rank1=rank1,
            window=self.window,
            verbose=0,
            n_iter=10000,
            n_jobs=-1,
        )
        self.cdl.raise_on_increase = False

    def run(self, cb):
        X = self.X
        if self.outliers_kwargs is not None:
            X = remove_outliers_before_cdl(self.X, self.z_shape, **self.outliers_kwargs)

        def alphacsc_cb(z_encoder, _):
            self.D_hat = z_encoder.D_hat
            cb()

        self.cdl.callback = alphacsc_cb
        self.cdl.fit(X)

    def get_result(self):
        if self.D_hat.shape[0] == 0:
            return {"D": self.D_init}
        return {"D": self.D_hat}
