"""This file contains the code to run the experiments for the outliers detection task
using the WinCDL algorithm. It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from torch import cuda

from wincdl.utils.utils_exp import (
    check_and_initialize,
    evaluate_D_hat,
    get_method_name,
    get_outliers_metric,
    plot_dicts,
)
from wincdl.utils.utils_signal import generate_experiment
from wincdl.wincdl import WinCDL

DEVICE = "cuda" if cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

EXP_DIR = Path("results")
EXP_DIR.mkdir(exist_ok=True, parents=True)

N_RUNS = 2
REG = 0.8


exp_dir = EXP_DIR / f"outlier_detection_reg_{REG}"
exp_dir.mkdir(exist_ok=True, parents=True)

# Base contamination parameters
contamination_params = {
    "n_atoms": 2,
    "sparsity": 3,
    "init_z": "constant",
    "init_z_kwargs": {"value": 50},
}

# Base simulation parameters
simulation_params = {
    "n_trials": 10,
    "n_channels": 2,
    "n_times": 5000,
    "n_atoms": 2,
    "n_atoms_extra": 2,
    "n_times_atom": 64,
    "window": True,
    "D_init": "random",
    "contamination_params": contamination_params,
    "init_d": "shapes",
    "init_d_kwargs": {"shapes": ["sin", "gaussian"]},
    "init_z": "constant",
    "init_z_kwargs": {"value": 1},
    "noise_std": 0.01,
    "rng": None,
    "sparsity": 20,
}
simulation_params["n_patterns_per_atom"] = simulation_params["n_channels"]

# Define base detection parameters
outliers_kwargs = dict(
    moving_average=None,
    union_channels=False,
    opening_window=True,
)
wincdl_params = {
    "n_components": 4,
    "kernel_size": 64,
    "n_channels": 2,
    "lmbd": REG,
    "scale_lmbd": True,
    "epochs": 30,
    "max_batch": 10,
    "mini_batch_size": 5,
    "mini_batch_window": 960,
    "optimizer": "linesearch",
    "n_iterations": 50,
    "stochastic": True,
    "window": True,
    "device": DEVICE,
}

df_results = check_and_initialize(
    exp_dir, simulation_params, wincdl_params, results_file_name="df_results"
)

# Compute the contamination percentage for the given parameters
_, _, _, _, info_contam = generate_experiment(
    simulation_params,
    return_info_contam=True,
)
percentage = info_contam["percentage"]

expected_D_shape = (
    wincdl_params["n_components"],
    wincdl_params["n_channels"],
    wincdl_params["kernel_size"],
)

# Define list of methods to test
alpha_true = round(percentage / 100, 1)
print(f"Percentage of contamination: {percentage}%, alpha_true: {alpha_true}")
list_methods = [
    None,
    {"method": "quantile", "alpha": alpha_true * 2},
    # {"method": "quantile", "alpha": alpha_true},
    # {"method": "quantile", "alpha": alpha_true / 2},
    # {"method": "iqr", "alpha": 1.5},
    # {"method": "zscore", "alpha": 1},
    # {"method": "zscore", "alpha": 2},
    # # {"method": "zscore", "alpha": 3},
    # # {"method": "mad", "alpha": 1},
    {"method": "mad", "alpha": 3.5},
]

if len(df_results) > 0:
    list_seeds = df_results["seed"].unique()
else:
    list_seeds = np.random.randint(0, 2**32 - 1, N_RUNS)


n_runs = len(list_seeds)
results = []


for i, seed in enumerate(list_seeds):
    print(f"Run {i+1}/{n_runs}")
    simulation_params["rng"] = seed

    # Default with random D_init
    X, _, D_true, D_init, info_contam = generate_experiment(
        simulation_params,
        D_init_shape=expected_D_shape,
        return_info_contam=True,
    )
    this_percentage = info_contam["percentage"]
    true_outliers_mask = info_contam["outliers_mask"].max(axis=1)

    if i == 0:
        # Plot true dictionary
        fig = plot_dicts(D_true)
        fig.savefig(exp_dir / "dict_true.pdf")

    # Update parameters
    wincdl_params["D_init"] = D_init

    for this_method in list_methods:

        method_name = get_method_name(this_method)
        if this_method is None:
            this_method = {}
            this_outliers_kwargs = None
        else:
            this_outliers_kwargs = dict(
                **outliers_kwargs, **this_method
            )

        def callback_fn(model, epoch, loss):
            if hasattr(model.loss_fn, "method"):
                metrics = get_outliers_metric(
                    true_outliers_mask, model, X
                )
            else:
                metrics = {}
            recovery_score = evaluate_D_hat(D_true, model.D_hat_)

            results.append({
                "name": method_name,
                **this_method, **info_contam, **metrics,
                "recovery_score": recovery_score,
                "seed": seed, "epoch": epoch, "loss": loss
            })


        # Run
        wincdl = WinCDL(
            **wincdl_params, outliers_kwargs=this_outliers_kwargs,
            callbacks=[callback_fn]
        )
        wincdl.fit(X)

        if i == 0:
            # Plot true dictionary
            fig = plot_dicts(wincdl.D_hat_)
            fig.savefig(exp_dir / f"D_hat_{method_name}.pdf")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(exp_dir / "df_results.csv", index=False)

curves = df_results.groupby(["name", "epoch"])["recovery_score"].mean()
for name in df_results.name.unique():
    curves.loc[name].plot(label=name)
plt.legend()
plt.savefig(exp_dir / "recovery_score.pdf")
