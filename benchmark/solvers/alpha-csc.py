from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    # import your reusable functions here
    from alphacsc.convolutional_dictionary_learning import BatchCDL, GreedyCDL
    from alphacsc.online_dictionary_learning import OnlineCDL

    ALGORITHMS = {
        "online": OnlineCDL,
        "batch": BatchCDL,
        "greedy": GreedyCDL,
    }


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = "alphaCSC"

    install_cmd = "conda"
    requirements = ["pip:alphacsc"]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {"type": ["online", "batch", "greedy"], "epochs": [100]}
    stopping_criterion = SufficientProgressCriterion(patience=5, strategy="callback")

    def get_next(self, stop_val):
        return stop_val + 1

    def set_objective(self, X, D_init, reg, window):
        self.X = X
        self.D_init = D_init
        self.reg = reg
        self.window = window

        # Rank1 dictionary if it is 2D
        rank1 = D_init.ndim == 2

        self.cdl = ALGORITHMS[self.type](
            n_atoms=D_init.shape[0],
            n_times_atom=D_init.shape[2],
            D_init=D_init,
            reg=reg,
            lmbd_max="scaled",
            solver_z="lgcd",
            rank1=rank1,
            window=self.window,
            verbose=0,
            n_iter=self.epochs,
        )
        self.cdl.raise_on_increase = False

    def run(self, cb):
        def alphacsc_cb(z_encoder, _):
            self.D_hat = z_encoder.D_hat
            cb()

        self.cdl.callback = alphacsc_cb
        self.cdl.fit(self.X)

    def get_result(self):
        if self.D_hat.shape[0] == 0:
            return {"D": self.D_init}
        return {"D": self.D_hat}
