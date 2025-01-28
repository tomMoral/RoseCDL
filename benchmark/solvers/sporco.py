from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np

    from sporco.dictlrn import cbpdndl
    from alphacsc.utils.dictionary import get_lambda_max
    from wincdl.utils.utils_outlier_comparison import remove_outliers_before_cdl


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Sporco'

    install_cmd = 'conda'
    requirements = ['pip:git+https://github.com/bwohlberg/sporco.git']

    parameters = {
        'outliers_kwargs': [
            None,
            {"method": "quantile", "alpha": 0.2},
            {"method": "iqr", "alpha": 1.5},
            {"method": "mad", "alpha": 3.5},
            {"method": "zscore", "alpha": 1.5},
        ],
    }

    stopping_criterion = SufficientProgressCriterion(patience=10)

    def get_next(self, stop_val):
        return stop_val + 1

    def skip(self, X, D_init, reg, window, has_outliers):
        if D_init.ndim == 2:
            return True, "Sporco only supports full rank dictionary"
        if not has_outliers and self.outliers_kwargs is not None:
            return True, "Don't run with outlier detection on data without outliers."

        return False, None

    def set_objective(self, X, D_init, reg, window, has_outliers):
        self.X = X
        self.D_init = D_init
        self.reg = reg
        self.has_outliers = has_outliers

        self.z_shape = tuple(
            xs - ds + 1 for xs, ds in zip(X.shape[2:], D_init.shape[2:])
        )
        self.z_shape = (X.shape[0], D_init.shape[0], *self.z_shape)

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

        X = self.X
        if self.outliers_kwargs is not None:
            X = remove_outliers_before_cdl(
                self.X, self.z_shape, **self.outliers_kwargs
            )
        reg = self.reg * get_lambda_max(X, self.D_init).max()

        sporco_params = dict(
            D0=self.D_init.transpose(2, 1, 0).copy(),
            S=X.transpose(2, 1, 0).copy(),
            lmbda=reg,
            opt=opt,
            dmethod="cns",
            dimN=len(self.D_init.shape[2:])
        )

        cdl = cbpdndl.ConvBPDNDictLearn(**sporco_params)
        cdl.solve()

        self.D = cdl.getdict()[:, :, 0, :].transpose(2, 1, 0)
        self.D /= np.linalg.norm(self.D, axis=(1, 2), keepdims=True)

    def get_result(self):
        return {"D": self.D}
