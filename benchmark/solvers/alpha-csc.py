from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    # import your reusable functions here
    from alphacsc.convolutional_dictionary_learning import BatchCDL, GreedyCDL
    from alphacsc.online_dictionary_learning import OnlineCDL


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'alphaCSC'

    install_cmd = 'conda'
    requirements = ['pip:alphacsc']

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'random_state': [42],
        'solver_z': ["lgcd"],
        'solver_d': ["fista"],
        'type': ["online", "batch"],
        # 'type': ["online", "batch", "greedy"],
    }
    stopping_criterion = SufficientProgressCriterion(
        patience=5, strategy='iteration'
    )

    def set_objective(
        self,
        X,
        D_init,
        reg,
        rank,
        window
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        self.X = X
        self.D_init = D_init
        self.rank = rank
        self.reg = reg
        self.window = window

        # Infer dictionary size from D_init
        self.n_atoms = D_init.shape[0]
        self.n_channels = self.X.shape[1]
        if self.rank == "full":
            self.kernel_size = D_init.shape[-1]
        else:
            self.kernel_size = D_init.shape[-1] - self.n_channels

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        self.n_channels = self.X.shape[1]

        if self.rank == "full":
            rank1 = False
            uv_constraint = 'auto'
        elif self.rank == "uv_constraint":
            rank1 = True
            uv_constraint = "separate"

        if self.type == "greedy":
            n_iter = self.n_atoms + n_iter

        alphacsc_params = dict(
            n_atoms=self.n_atoms,
            n_times_atom=self.kernel_size,
            random_state=self.random_state,
            rank1=rank1,
            lmbd_max="fixed",
            n_iter=n_iter,
            window=self.window,
            verbose=0,
            solver_z=self.solver_z,
            solver_d=self.solver_d,
            uv_constraint=uv_constraint
        )

        if self.type == "online":
            cdl = OnlineCDL(
                **alphacsc_params,
                D_init=self.D_init,
                reg=self.reg
            )
        if self.type == "batch":
            cdl = BatchCDL(
                **alphacsc_params,
                D_init=self.D_init,
                reg=self.reg
            )
        if self.type == "greedy":
            cdl = GreedyCDL(
                **alphacsc_params,
                D_init=self.D_init,
                reg=self.reg
            )
        cdl.fit(self.X)

        self.D = cdl.D_hat_

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return {"D": self.D}
