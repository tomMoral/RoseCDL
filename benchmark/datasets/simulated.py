from benchopt import BaseDataset, safe_import_context


# Allow installing the requirements with `benchopt install`
with safe_import_context() as import_ctx:
    from alphacsc.init_dict import init_dictionary
    from wincdl.datasets.simulated import simulate_1d


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_times': [
            (10, 5000),
        ],
        'random_state': [27],
    }

    def get_data(self):
        X, D, _ = simulate_1d(
            n_atoms=10, n_times_atom=50, n_trials=self.n_samples,
            n_times=self.n_times, random_state=self.random_state
        )
        X, D = X[:, None], D[:, None]

        D_init = init_dictionary(X, n_atoms=15, n_times_atom=50, rank1=False,
                        window=True)



        return dict(
            X=X, D=D, D_init=D_init, rank="full", window=True
        )
