from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from alphacsc.utils.convolution import construct_X_multi
    from alphacsc.loss_and_gradient import compute_objective
    from alphacsc.update_z_multi import update_z_multi
    from wincdl.utils.utils_exp import evaluate_D_hat


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Convolutional Dictionary Learning"

    # URL of the main repo for this benchmark.
    url = "https://github.com/tommoral/WinCDL"


    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.6"

    install_cmd = 'conda'
    requirements = [
        'pip:alphacsc', 'pip:git+https://github.com/tommoral/WinCDL#egg=wincdl'
    ]

    parameters = {
        'reg': [8e-1],
    }

    def set_data(self, X, D, D_init, rank, window):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.D = X, D
        self.D_init = D_init
        self.rank, self.window = rank, window
        self.z0_ = None


    def evaluate_result(self, D):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        z_hat, _, _ = update_z_multi(
            self.X, D.astype(float), reg=self.reg, z0=self.z0_,
            solver="lgcd", solver_kwargs={'tol': 1e-3}, n_jobs=4
        )
        X_hat = construct_X_multi(z_hat, D=D)
        cost = compute_objective(
            self.X, X_hat=X_hat, z_hat=z_hat, D=D, reg=self.reg
        )
        # Warm start for next computation
        self.z0_ = z_hat

        # Evaluate recovery of the dictionary
        if self.D is not None:
            recovery = evaluate_D_hat(self.D, D)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=cost,
            recovery=recovery
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(D=self.D_init)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            X=self.X,
            D_init=self.D_init,
            reg=self.reg,
            rank=self.rank,
            window=self.window
        )
