from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch

    # import your reusable functions here
    from wincdl.wincdl import WinCDL


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = "WinCDL"

    install_cmd = "conda"
    requirements = ["pip:torch", "pip:tqdm"]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "epochs": [100],
        "n_csc_iterations": [25],
        "mini_batch_size": [1],
        "sample_window": [1000],
        "random_state": [42],
        "outliers_kwargs": [dict(method="quantile", alpha=0.05)],
    }

    stopping_criterion = SufficientProgressCriterion(patience=5, strategy="callback")

    def get_next(self, stop_val):
        return stop_val + 1

    def set_objective(
        self,
        X,
        D_init,
        reg,
        window,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        self.X, self.reg, self.D_init = X, reg, D_init
        self.window = window

        # Infer dictionary size from D_init
        self.n_atoms = D_init.shape[0]
        self.n_channels = self.X.shape[1]
        self.atom_support = D_init.shape[2:]

        rank1 = D_init.ndim == 2

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_kwargs = dict(
            lmbd=self.reg,
            D_init=self.D_init,
            window=self.window,
            rank1=rank1,
            n_iterations=self.n_csc_iterations,
            optimizer="linesearch",
            mini_batch_size=10,
            sample_window=self.sample_window,
            max_batch=10,
            epochs=self.epochs,
            outliers_kwargs=self.outliers_kwargs,
            device=self.device,
            random_state=self.random_state,
        )

    def run(self, cb):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        self.model = WinCDL(**self.model_kwargs, callbacks=[lambda *x: not cb()])
        cb()  # Get init value
        self.model.fit(self.X)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return {"D": self.model.D_hat_}
