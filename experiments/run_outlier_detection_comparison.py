"""This file contains the code to run the experiments for the outliers detection task
using the WinCDL algorithm. It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from torch import cuda
from tqdm import tqdm

from wincdl.utils.utils_exp import (
    evaluate_D_hat,
    get_method_name,
    get_outliers_metric,
    plot_dicts,
)
from wincdl.utils.utils_signal import generate_experiment
from wincdl.wincdl import WinCDL

mem = Memory(location="__cache__", verbose=0)


def allowed_detection_timings(
    cdl_package: str, outlier_detection_timing_list: list[str]
):
    if cdl_package == "wincdl":
        return outlier_detection_timing_list
    return [
        timing for timing in outlier_detection_timing_list if timing != "during"
    ]


def allowed_detection_methods(
    outlier_detection_timing: str, outlier_detection_method_list: list[str]
) -> list[str]:
    if outlier_detection_timing == "never":
        return [
            method
            for method in outlier_detection_method_list
            if method["name"] == "none"
        ]
    return [
        method
        for method in outlier_detection_method_list
        if method["name"] != "none"
    ]


def generate_run_config_list(
    cdl_package_list: list,
    outlier_detection_method_list: list,
    outlier_detection_timing_list: list,
):
    run_config_list = []
    for package in cdl_package_list:
        for timing in allowed_detection_timings(
            package, outlier_detection_timing_list
        ):
            for method in allowed_detection_methods(
                timing, outlier_detection_method_list
            ):
                run_config_list.append(
                    {"package": package, "method": method, "timing": timing}
                )
    return run_config_list


@mem.cache
def run_one(
    cdl_package: str,
    outlier_detection_method: dict[str, str or float],
    outlier_detection_timing: str,
    outliers_kwargs: dict[str, str or float],
    cdl_params: dict[str, str or float],
    simulation_params: dict[str, str or float],
    seed: float,
    i: int,
):
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        cdl_package (str): CDL package name: "wincdl", "alphacsc" or "sporco".
        outlier_detection_method (str): Outlier detection method.
            A dictionary is expected with two keys:
                - "name": name of the method. Can be one of "quantile", "iqr",
                    "zscore", "mad" or "none".
                - "alpha" (float): Parameter of the method.
        outlier_detection_timing (str): When to run the outlier detection
            relatively to the CDL: "before", "during" or "never".
        outliers_kwargs (dict): Additional parameters for outlier detection.
        cdl_params (dict): Parameters for the CDL algorithm.
        simulation_params (dict): Parameters for data simulation.
        seed (float): Random seed.
        i (int): Counting index of the run.

    """

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
        this_outliers_kwargs = dict(**outliers_kwargs, **method)

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

        results.append(
            {
                "name": method_name,
                **method,
                **info_contam,
                **metrics,
                "recovery_score": recovery_score,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
            }
        )

    # Run the experiment
    wincdl = WinCDL(
        **wincdl_params,
        D_init=D_init,
        outliers_kwargs=this_outliers_kwargs,
        callbacks=[callback_fn],
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
        description="Run a comparison of the different methods for dictionary recovery"
    )
    parser.add_argument(
        "--n-jobs", "-j", type=int, default=1, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Master seed for reproducible job",
    )
    parser.add_argument(
        "--n-runs",
        "-n",
        type=int,
        default=20,
        help="Number of repetitions for the experiment",
    )
    parser.add_argument(
        "--reg", type=float, default=0.8, help="Regularization parameter"
    )
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
        "sample_window": 960,
        "optimizer": "linesearch",
        "n_iterations": 50,
        "window": True,
        "device": DEVICE,
    }
    alphacsc_params = {}
    sporco_params = {}

    cdl_package_list = ["wincdl", "alphacsc", "sporco"]
    outlier_detection_method_list = [
        {"method": "none", "alpha": -1},
        {"method": "quantile", "alpha": 0.05},
        {"method": "quantile", "alpha": 0.1},
        {"method": "quantile", "alpha": 0.2},
        {"method": "iqr", "alpha": 1.5},
        {"method": "zscore", "alpha": 1},
        {"method": "zscore", "alpha": 2},
        {"method": "mad", "alpha": 3.5},
    ]
    outlier_detection_timing_list = ["before", "during", "never"]

    run_config_list = generate_run_config_list(
        cdl_package_list=cdl_package_list,
        outlier_detection_method_list=outlier_detection_method_list,
        outlier_detection_timing_list=outlier_detection_timing_list,
    )
    for run_config in run_config_list:
        run_config["cdl_params"] = {
            "alphacsc": alphacsc_params,
            "sporco": sporco_params,
            "wincdl": wincdl_params,
        }[run_config["package"]]

    for run_config in run_config_list:
        pass

    list_seeds = rng.integers(0, 2**32 - 1, n_runs)
    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            cdl_package=run_config["package"],
            outlier_detection_method=run_config["method"],
            outlier_detection_timing=run_config["timing"],
            outliers_kwargs=outliers_kwargs,
            cdl_params=run_config["cdl_params"],
            simulation_params=simulation_params,
            seed=seed,
            i=i,
        )
        for i, seed in enumerate(list_seeds)
        for run_config in run_config_list
    )
    results = list(
        r
        for res in tqdm(results, "Running", total=n_runs * len(run_config_list))
        for r in res
    )

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    curves = df_results.groupby(["name", "epoch"])["recovery_score"].mean()
    for name in df_results.name.unique():
        curves.loc[name].plot(label=name)
    plt.legend()
    plt.savefig(exp_dir / "recovery_score.pdf")
