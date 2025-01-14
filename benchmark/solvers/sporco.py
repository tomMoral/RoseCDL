from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    # import your reusable functions here
    from sporco.dictlrn import cbpdndl

    import numpy as np


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Sporco'

    install_cmd = 'conda'
    requirements = ['pip:sporco']

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'random_state': [42],
        # 'xmethod': ["adam"],
        # 'dmethod': ["pgm"],
        'type': ["BPDN"],
    }

    def skip(
        self,
        X,
        D_init,
        reg,
        rank,
        window
    ):

        if rank != "full":
            return True, "Sporco only supports full rank dictionary"
        # if window:
        #     return True, "Sporco does not support windowed atoms"

        return False, None

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
        self.reg = reg

    def run(self, n_iter):

        self.n_channels = self.X.shape[1]

        opt_cbpdn = cbpdndl.ConvBPDNOptionsDefaults()

        opt_cbpdn["NonNegCoef"] = True
        opt_cbpdn["Verbose"] = False
        opt_cbpdn["AuxVarObj"] = False
        # opt_cbpdn["FastSolve"] = True

        opt = cbpdndl.ConvBPDNDictLearn.Options({
            'Verbose': False, 'MaxMainIter': n_iter + 1, 'CBPDN': opt_cbpdn
        }, dmethod="cns")

        sporco_params = dict(
            D0=self.D_init.transpose(2, 1, 0).copy(),
            S=self.X.transpose(2, 1, 0).copy(),
            lmbda=self.reg,
            opt=opt,
            dmethod="cns",
            dimN=1
        )

        cdl = cbpdndl.ConvBPDNDictLearn(**sporco_params)
        cdl.solve()

        self.D = cdl.getdict()[:, :, 0, :].transpose(2, 1, 0)
        self.D /= np.linalg.norm(self.D, axis=(1, 2), keepdims=True)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return {"D": self.D}
