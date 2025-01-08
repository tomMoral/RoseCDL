"""This file contains the code to run the experiments for the outliers detection task
using the WinCDL algorithm. It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from torch import cuda
from tqdm import tqdm

from wincdl.utils.utils_exp import (
    check_and_initialize,
    evaluate_D_hat,
    get_lambda_max,
    get_method_name,
    get_outliers_metric,
    load_json,
    plot_dicts,
)
from wincdl.utils.utils_signal import generate_experiment
from wincdl.wincdl import WinCDL

DEVICE = "cuda" if cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

EXP_DIR = Path("outliers_detection")
EXP_DIR.mkdir(exist_ok=True, parents=True)

add_reg = True
per_patch = True
q_lmnd = "method"
REG = 0.8

D_INIT = "random"  # "random" or "chunk"

n_runs = 2

exp_name = (
    f"add_reg_{add_reg}_per_patch_{per_patch}_q_{q_lmnd}_reg_{REG}_D_init_{D_INIT}"
)
exp_dir = EXP_DIR / exp_name
exp_dir.mkdir(exist_ok=True, parents=True)

# Load parameters from json files
simulation_params = load_json(Path("outliers_detection/simulation_params.json"))
wincdl_params = load_json(Path("outliers_detection/wincdl_params.json"))

# Base contamination parameters
contamination_params = {
    "n_atoms": 2,
    "sparsity": 3,
    "init_z": "constant",
    "init_z_kwargs": {"value": 50},
}

# Base simulation parameters
simulation_params.update(
    dict(
        contamination_params=contamination_params,
        D_init=D_INIT,
        # Simulate with a pattern on each channel
        n_patterns_per_atom=simulation_params["n_channels"],
    )
)

# Define base detection parameters
moving_average = dict(
    window_size=int(wincdl_params["kernel_size"] / 2),
    method="average",  # 'max', 'average' or 'gaussian'
)
outliers_kwargs = dict(
    moving_average=None,
    union_channels=False,
    add_reg=add_reg,
    opening_window=True,
    per_patch=per_patch,
)
wincdl_params.update(
    dict(
        outliers_kwargs=outliers_kwargs,
        device=DEVICE,
        epochs=30,  # previously 100
    )
)

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
    {"method": "quantile_unilateral", "alpha": alpha_true * 2},
    {"method": "quantile_unilateral", "alpha": alpha_true},
    {"method": "quantile_unilateral", "alpha": alpha_true / 2},
    {"method": "iqr_unilateral", "alpha": 1.5},
    {"method": "zscore", "alpha": 1},
    {"method": "zscore", "alpha": 2},
    # {"method": "zscore", "alpha": 3},
    # {"method": "mad", "alpha": 1},
    {"method": "mad", "alpha": 3.5},
]

if len(df_results) > 0:
    list_seeds = df_results["seed"].unique()
else:
    list_seeds = np.random.randint(0, 2**32 - 1, n_runs)

# Initialize dictionaries to save D_hat
dict_D_hat = dict()
for this_method in list_methods:
    method_name = get_method_name(this_method)
    dict_D_hat[method_name] = []

n_runs = len(list_seeds)

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
    true_outliers_mask = info_contam["outliers_mask"]

    if i == 0:
        # Plot true dictionary
        fig = plot_dicts(D_true)
        fig.savefig(exp_dir / "dict_true.pdf")

    # Update parameters
    wincdl_params["D_init"] = D_init

    for this_method in list_methods:
        if this_method is None:
            this_outliers_kwargs = None
            lmbd_max = get_lambda_max(X, D_init).max()
        else:
            outliers_kwargs.update(**this_method)
            this_outliers_kwargs = outliers_kwargs

            q_method = outliers_kwargs["method"]
            if "quantile" in q_method:
                q = 1 - outliers_kwargs["alpha"]
                q_method = "quantile"
            else:
                q = outliers_kwargs["alpha"]

            if "iqr" in q_method:
                q_method = "iqr"

            lmbd_max = get_lambda_max(X, D_init, q=q, method=q_method).max()

        wincdl_params.update(
            dict(outliers_kwargs=this_outliers_kwargs, lmbd=REG * lmbd_max)
        )

        method_name = get_method_name(this_method)

        # Run
        wincdl = WinCDL(**wincdl_params)
        losses, list_D, times = wincdl.fit(X)

        recovery_scores = [evaluate_D_hat(D_true, this_D) for this_D in list_D]

        # Save D_hat
        dict_D_hat[method_name].append(list_D[-1])

        if this_method is not None:
            acc, prec, recall, f1, percent, dice, jaccard = [], [], [], [], [], [], []
            print("Computing metrics...")
            for this_D in tqdm(list_D):
                metrics, outliers_mask = get_outliers_metric(
                    wincdl,
                    true_outliers_mask,
                    X,
                    this_D,
                    return_mask=True,
                    with_opening_window=False,
                )
                acc.append(metrics["accuracy"])
                prec.append(metrics["precision"])
                recall.append(metrics["recall"])
                f1.append(metrics["f1"])
                dice.append(metrics["dice"])
                jaccard.append(metrics["jaccard"])
                percent.append(metrics["percentage"])
        else:
            acc, prec, recall, f1, dice, jaccard, percent = (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # Save results
        df_temp = pd.DataFrame()
        df_temp["loss"] = losses
        df_temp["times"] = np.cumsum(times)
        df_temp["score"] = recovery_scores
        df_temp["iteration"] = np.arange(wincdl_params["epochs"] + 1)
        df_temp["seed"] = seed
        df_temp["init"] = simulation_params["D_init"]
        df_temp["true_percentage"] = percentage
        df_temp["current_percentage"] = this_percentage
        df_temp["effective_percentage"] = percent
        df_temp["method"] = method_name
        df_temp["accuracy"] = acc
        df_temp["precision"] = prec
        df_temp["recall"] = recall
        df_temp["f1"] = f1
        df_temp["dice"] = dice
        df_temp["jaccard"] = jaccard
        df_temp["lmbd_max"] = lmbd_max
        df_temp["reg"] = REG
        df_results = pd.concat([df_results, df_temp], ignore_index=True)

        print(f"Method {method_name}, Final score: {recovery_scores[-1]:.5f}")

    df_results.to_csv(exp_dir / "df_results.csv", index=False)

# Save list of D_hat
np.savez(exp_dir / "dict_D_hat.npz", **dict_D_hat)

dict_D_hat = np.load(exp_dir / "dict_D_hat.npz", allow_pickle=True)
# Plot D_hat per method
for method_name in dict_D_hat:
    fig = plot_dicts(*dict_D_hat[method_name])
    fig.savefig(exp_dir / f"D_hat_{method_name}.pdf")
