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

from wincdl.loss import LassoLoss, OutlierLoss
from wincdl.utils.utils_exp import (
    evaluate_D_hat,
    get_method_name,
    get_outliers_metric,
    plot_dicts,
)
from wincdl.utils.utils_signal import generate_experiment
from wincdl.wincdl import WinCDL

EXP_DIR = Path.home() / "data/wincdl"

mem = Memory(location="__cache__", verbose=0)


def remove_outliers_before_cdl(
    data: np.array,
    activation_vector_shape: tuple,
    lmbd: float,
    method_spec: dict[str, str or float],
    outliers_kwargs: dict[str, str or bool],
) -> np.array:
    """Remove outliers before CDL.

    Args:
        data: array of shape (n_trials, n_channels, n_times)
        method_spec: dict with two keys ("name" and "alpha") that specifies
            the outlier detection method
        outlier_kwargs: additional arguments for outlier detection
            (moving_average, opening_window, union_channels)
    """
    lasso_loss = LassoLoss(lmbd=lmbd, reduction="sum")
    outlier_loss = OutlierLoss(
        lasso_loss,
        method=method_spec["name"],
        alpha=method_spec["alpha"],
        moving_average=outliers_kwargs["moving_average"],
        opening_window=outliers_kwargs["opening_window"],
        union_channels=outliers_kwargs["union_channels"],
    )

    outlier_mask = outlier_loss.get_outliers_mask(
        X_hat=np.zeros_like(data),
        z_hat=np.zeros(activation_vector_shape),
        X=data,
    )

    data[outlier_mask] = data.mean()
    return data


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
    seed: int,
    i: int,
    exp_dir: str,
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
        seed (int): Random seed.
        i (int): Counting index of the run.
        exp_dir (str): Name of the directory to store the results

    """

    # Generate the data
    simulation_params["rng"] = seed
    X, z, D_true, D_init, info_contam = generate_experiment(
        simulation_params,
        return_info_contam=True,
    )

    if i == 0:
        # Plot true dictionary
        fig = plot_dicts(D_true)
        fig.savefig(exp_dir / "dict_true.pdf")

    # retrieve the method
    method_name = get_method_name(outlier_detection_method)
    if outlier_detection_method["name"] == "name":
        summary_method = {}
    else:
        summary_method = outlier_detection_method

    # Parameters for WinCDL's outlier loss.
    # (only for cdl_package="wincdl" and outlier_detection_timing="during")
    online_outliers_kwargs = {
        "never": {},
        "before": {},
        "during": {**outliers_kwargs, **outlier_detection_method},
    }[outlier_detection_timing]

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
                **summary_method,
                **info_contam,
                **metrics,
                "recovery_score": recovery_score,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
            }
        )

    # Perform outlier detection on the data before CDL
    if outlier_detection_timing == "before":
        X = remove_outliers_before_cdl(
            data=X,
            activation_vector_shape=z.shape,
            lmbd=cdl_params["lmbd"],
            method_spec=outlier_detection_method,
            outliers_kwargs=outliers_kwargs,
        )

    # Run the experiment
    if cdl_package == "wincdl":
        wincdl = WinCDL(
            **cdl_params,
            D_init=D_init,
            outliers_kwargs=online_outliers_kwargs,
            callbacks=[callback_fn],
        )
        wincdl.fit(X)
    elif cdl_package == "alphacsc":
        return []
    elif cdl_package == "sporco":
        return []
    else:
        raise ValueError(f"Unknown CDL package {cdl_package}")

    # Plot true dictionary
    if i == 0:
        fig = plot_dicts(wincdl.D_hat_)
        fig.savefig(exp_dir / f"D_hat_{method_name}.pdf")

    # Save the results
    df_results = pd.DataFrame(results)
    file_name = (
        f"{cdl_package}_"
        f"{method_name.replace(' ','_')}_"
        f"{outlier_detection_timing}_"
        f"seed_{seed}.csv"
    )
    df_results.to_csv(exp_dir / file_name, index=False)

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
        {"name": "none", "alpha": -1},
        {"name": "quantile", "alpha": 0.05},
        {"name": "quantile", "alpha": 0.1},
        {"name": "quantile", "alpha": 0.2},
        {"name": "iqr", "alpha": 1.5},
        {"name": "zscore", "alpha": 1},
        {"name": "zscore", "alpha": 2},
        {"name": "mad", "alpha": 3.5},
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
            exp_dir=exp_dir,
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
