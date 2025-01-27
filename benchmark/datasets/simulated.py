from benchopt import BaseDataset, safe_import_context

# Allow installing the requirements with `benchopt install`
with safe_import_context() as import_ctx:
    from wincdl.utils.utils_signal import generate_experiment


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        "n_samples, n_times": [
            (10, 5000),
        ],
        # shape of the searched dictionary
        "n_atoms, n_times_atom": [(5, 64)],
        "random_state": [None],
        "contamination": [0.1],
    }
    parameter_template = "T={n_times},outliers={contamination}"

    test_parameters = {
        "n_samples, n_times, n_atoms, n_times_atom": [(1, 250, 1, 10)]
    }

    def get_data(self):
        size = self.n_times // 5000
        contamination_params = {
            "n_atoms": 2 * size,
            "sparsity": 3 * size,
            "init_z": "constant",
            "init_z_kwargs": {"value": 50},
        } if self.has_outliers else None

        simulation_params = {
            'n_trials': self.n_samples,
            'n_channels': 2,
            'n_times': self.n_times,
            'n_atoms': self.n_atoms,
            'n_times_atom': self.n_times_atom,
            'n_atoms_extra': 5,  # extra atoms in the learned dictionary
            'D_init': "random",
            'window': True,
            'contamination_params': contamination_params,
            'init_d': "shapes",
            'init_d_kwargs': {"shapes": ["sin", "gaussian"]},
            'init_z': "constant",
            'init_z_kwargs': {"value": 1},
            'noise_std': 0.01,
            'rng': self.random_state,
            'sparsity': 20 * size,
        }
        X, _, D, D_init, info_contam = generate_experiment(
            simulation_params=simulation_params, return_info_contam=True,
        )
        outliers = info_contam.get("outliers_mask", None)

        return dict(X=X, D=D, D_init=D_init, outliers=outliers, window=True)
