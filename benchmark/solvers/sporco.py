from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

import numpy as np
from alphacsc.utils.dictionary import get_lambda_max
from sporco.dictlrn import cbpdndl

from rosecdl.utils.utils_outlier_comparison import remove_outliers_before_cdl


class Solver(BaseSolver):
    name = "Sporco"

    install_cmd = "conda"
    requirements = ["pip::git+https://github.com/bwohlberg/sporco.git"]

    parameters = {
        "outliers_kwargs": [
            None,
            {"method": "mad", "alpha": 3.5},
        ],
    }

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="callback"
    )

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
            xs - ds + 1 for xs, ds in zip(X.shape[2:], D_init.shape[2:], strict=False)
        )
        self.z_shape = (X.shape[0], D_init.shape[0], *self.z_shape)

    def run(self, cb):
        self.n_channels = self.X.shape[1]

        def callback_fn(model, *args):
            self.D = cdl.getdict()
            return not cb()

        opt_cbpdn = cbpdndl.ConvBPDNOptionsDefaults()

        opt_cbpdn["NonNegCoef"] = True
        opt_cbpdn["Verbose"] = False
        opt_cbpdn["AuxVarObj"] = False
        # opt_cbpdn["FastSolve"] = True

        opt = cbpdndl.ConvBPDNDictLearn.Options(
            {
                "Verbose": False,
                "MaxMainIter": 10000,
                "Callback": callback_fn,
                "CBPDN": opt_cbpdn},
            dmethod="cns",
        )

        X = self.X
        if self.outliers_kwargs is not None:
            X = remove_outliers_before_cdl(self.X, self.z_shape, **self.outliers_kwargs)

        sporco_params = dict(
            D0=self.D_init.transpose(2, 1, 0).copy(),
            S=X.transpose(2, 1, 0).copy(),
            lmbda=self.reg,
            opt=opt,
            dmethod="cns",
            dimN=len(self.D_init.shape[2:]),
        )

        cdl = cbpdndl.ConvBPDNDictLearn(**sporco_params)
        cdl.solve()

        self.D = cdl.getdict()

    def get_result(self):
        D = self.D[:, :, 0, :].transpose(2, 1, 0).copy()
        D /= np.linalg.norm(self.D, axis=(1, 2), keepdims=True)
        return {"D": self.D}
