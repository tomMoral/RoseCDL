"""This file contains the code to run the experiments for the outliers detection task
using the WinCDL algorithm. It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from torch import cuda
from tqdm import tqdm
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from joblib import Memory

from wincdl.utils.utils_exp import (
    evaluate_D_hat,
    get_method_name,
    get_outliers_metric,
    plot_dicts,
)
from wincdl.utils.utils_signal import generate_experiment
from wincdl.wincdl import WinCDL


mem = Memory(location="__cache__", verbose=0)


EXP_DIR = Path("results")
EXP_DIR.mkdir(exist_ok=True, parents=True)


@mem.cache
def run_one(method, outliers_kwargs, wincdl_params, simulation_params, i, seed, exp_dir):

    # Generate the data
    simulation_params["rng"] = seed
    X, _, D_true, D_init, info_contam = generate_experiment(
        simulation_params,
        return_info_contam=True,
    )

    if i == 0:
        # Plot true dictionary
        fig = plot_dicts(D_true)
        fig.savefig(exp_dir / "dict_true.pdf")

    # retrieve the method
    method_name = get_method_name(method)
    if method is None:
        method = {}
        this_outliers_kwargs = None
    else:
        this_outliers_kwargs = dict(
            **outliers_kwargs, **method
        )

    # Setup the callback
    results = []
    def callback_fn(model, epoch, loss):
        if hasattr(model.loss_fn, "method"):
            metrics = get_outliers_metric(
                info_contam["outliers_mask"].max(axis=1), model, X
            )
        else:
            metrics = {}
        recovery_score = evaluate_D_hat(D_true, model.D_hat_)

        results.append({
            "name": method_name,
            **method, **info_contam, **metrics,
            "recovery_score": recovery_score,
            "seed": seed, "epoch": epoch, "loss": loss
        })

    # Run the experiment
    wincdl = WinCDL(
        **wincdl_params, D_init=D_init, outliers_kwargs=this_outliers_kwargs,
        callbacks=[callback_fn]
    )
    wincdl.fit(X)

    # Plot true dictionary
    if i == 0:
        fig = plot_dicts(wincdl.D_hat_)
        fig.savefig(exp_dir / f"D_hat_{method_name}.pdf")

    return results


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Run a comparison of the different methods for dictionary recovery'
    )
    parser.add_argument('--n-jobs', '-j', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Master seed for reproducible job')
    parser.add_argument('--n-runs', '-n', type=int, default=20,
                        help='Number of repetitions for the experiment')
    parser.add_argument('--reg', type=float, default=0.8,
                        help='Regularization parameter')
    args = parser.parse_args()

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")
    rng = np.random.default_rng(seed)

    n_runs = args.n_runs
    reg = args.reg


    exp_dir = EXP_DIR / f"outlier_detection_reg_{reg}"
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
        "n_atoms_extra": 2,  # extra atoms in the learned dictionary
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
        "lmbd": reg,
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

    # Define list of methods to test
    alpha_true = 0.1
    list_methods = [
        None,
        {"method": "quantile", "alpha": alpha_true * 2},
        {"method": "quantile", "alpha": alpha_true},
        {"method": "quantile", "alpha": alpha_true / 2},
        {"method": "iqr", "alpha": 1.5},
        {"method": "zscore", "alpha": 1},
        {"method": "zscore", "alpha": 2},
        # {"method": "zscore", "alpha": 3},
        # {"method": "mad", "alpha": 1},
        {"method": "mad", "alpha": 3.5},
    ]

    list_seeds = rng.integers(0, 2**32 - 1, n_runs)
    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            this_method, outliers_kwargs, wincdl_params, simulation_params, i, seed, exp_dir
        )
        for i, seed in enumerate(list_seeds)
        for this_method in list_methods
    )
    results = list(r for res in tqdm(results, "Running", total=n_runs*len(list_methods)) for r in res)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    curves = df_results.groupby(["name", "epoch"])["recovery_score"].mean()
    for name in df_results.name.unique():
        curves.loc[name].plot(label=name)
    plt.legend()
    plt.savefig(exp_dir / "recovery_score.pdf")
