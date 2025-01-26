from benchopt import BaseDataset, safe_import_context

# Allow installing the requirements with `benchopt install`
with safe_import_context() as import_ctx:
    import numpy as np
    from alphacsc.init_dict import init_dictionary

    from wincdl.datasets.simulated import simulate_1d


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        "n_samples, n_times": [
            (10, 10000),
        ],
        # shape of the searched dictionary
        "n_atoms, n_times_atom": [(15, 64)],
        "random_state": [None],
        "contamination": [0.1],
    }

    test_parameters = {
        "n_samples, n_times, n_atoms, n_times_atom": [(1, 250, 1, 10, 0.1)]
    }

    def get_data(self):
        rng = np.random.default_rng(self.random_state)
        X, D, _, outliers = simulate_1d(
            n_atoms=10,
            n_times_atom=50,
            n_trials=self.n_samples,
            n_times=self.n_times,
            p_acti=0.7,
            p_contaminate=self.contamination,
            random_state=rng.integers(0, 2**32),
        )
        # Add one channel dimension
        X, D = X[:, None], D[:, None]

        D_init = init_dictionary(
            X,
            n_atoms=self.n_atoms,
            n_times_atom=self.n_times_atom,
            rank1=False,
            window=True,
            D_init="random",
            random_state=rng.integers(0, 2**32),
        )

        return dict(X=X, D=D, D_init=D_init, outliers=outliers, window=True)
