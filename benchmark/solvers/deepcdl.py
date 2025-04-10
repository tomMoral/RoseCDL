from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch

    # import your reusable functions here
    from rosecdl.rosecdl import RoseCDL


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "DeepCDL"

    requirements = [
        "pip:torch",
        "pip:git+https://github.com/tomMoral/RoseCDL.git"
    ]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        # sample_window is defined as a multiple of the atom_support
        "mini_batch_size": [1],
        "sample_window": [10, 20, 50],
        "n_csc_iterations": [50],
        "random_state": [None],
    }

    test_parameters = {
        "mini_batch_size": [1],
        "sample_window": [10],
        "n_csc_iterations": [10],
        "random_state": [None],
    }

    stopping_criterion = SufficientProgressCriterion(patience=15, strategy="callback")

    def get_next(self, stop_val):
        return stop_val + 3

    def set_objective(
        self,
        X,
        D_init,
        reg,
        window,
        has_outliers,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

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
            scale_lmbd=True,
            D_init=torch.tensor(self.D_init),
            window=self.window,
            rank1=rank1,
            outliers_kwargs=None,
            epochs=10000,
            max_batch=None,
            mini_batch_size=self.mini_batch_size,
            sample_window=sample_window,
            deepcdl=True,
            optimizer="linesearch",
            n_iterations=self.n_csc_iterations,
            random_state=self.random_state,
            device=self.device,
        )

    def run(self, cb):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        self.model = RoseCDL(**self.model_kwargs, callbacks=[lambda *x: not cb()])
        cb()  # Get init value
        self.model.fit(self.X)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return {"D": self.model.D_hat_}
