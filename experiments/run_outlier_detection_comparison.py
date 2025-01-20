"""This file contains the code to run the experiments for the outliers detection task
using the WinCDL algorithm. It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alphacsc import BatchCDL
from joblib import Memory, Parallel, delayed
from torch import cuda
from tqdm import tqdm

from wincdl.utils.utils_exp import evaluate_D_hat, get_outliers_metric, plot_dicts
from wincdl.utils.utils_signal import generate_experiment
from wincdl.utils_outlier_comparison import remove_outliers_before_cdl
from wincdl.wincdl import WinCDL

mem = Memory(location="__cache__", verbose=0)


def generate_run_config_list(
    cdl_packages,
    outlier_detection_methods,
    outlier_detection_timings,
    cdl_configs,
    n_runs=1,
    seed=None,
):
    """Generate the list of configurations for the experiment.

    Args:
        cdl_packages (list): List of CDL packages to use.
        outlier_detection_methods (list): List of outlier detection methods.
        outlier_detection_timings (list): List of outlier detection timings.
        cdl_configs (dict): Dictionary of CDL configurations.
        n_runs (int): Number of runs to generate.
        seed (int): Master seed for the experiment.
    """
    # Generate a list of seeds for reproducibility
    rng = np.random.default_rng(seed)
    list_seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    run_config_list = []
    for package in cdl_packages:
        for timing in outlier_detection_timings:
            if package != "wincdl" and timing == "during":
                # Only wincdl can run the outlier detection during the CDL
                continue
            for method in outlier_detection_methods:
                if bool(timing == "never") != bool(method["method"] == "none"):
                    # XOR condition either timing is "never" and method is "none"
                    # or timing is not "never" and method is not "none"
                    continue
                run_config_list.extend(
                    {
                        "cdl_package": package,
                        "cdl_params": cdl_configs[package],
                        "outlier_detection_method": method,
                        "outlier_detection_timing": timing,
                        "seed": s,
                        "i": i,
                    }
                    for i, s in enumerate(list_seeds)
                )
    return run_config_list


@mem.cache
def run_one(
    cdl_package: str,
    cdl_params: dict[str, str or float],
    outlier_detection_method: dict[str, str or float],
    outlier_detection_timing: str,
    seed: int,
    i: int,
    outliers_kwargs: dict[str, str or float],
    simulation_params: dict[str, str or float],
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
    if cdl_package == "sporco":
        return []

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

    # Process the outlier method's parameters and compute the summary name
    if outlier_detection_timing == "never":
        summary_name, outliers_kwargs = "no detection", None
        outlier_detection_method = {}
    else:
        summary_name = "{method} (alpha={alpha:.02f})".format(
            **outlier_detection_method
        )
        outliers_kwargs = {**outliers_kwargs, **outlier_detection_method}

    summary_name = f"[{cdl_package}] {summary_name}"
    if outlier_detection_timing != "never":
        summary_name += f" ({outlier_detection_timing})"

    # Perform outlier detection on the data before CDL
    if outlier_detection_timing == "before":
        X = remove_outliers_before_cdl(
            data=X,
            activation_vector_shape=z.shape,
            **outliers_kwargs,
        )
        outliers_kwargs = None

    # Setup the callback
    results = []

    def callback_fn(model, *args):
        if len(args) == 1:
            pobj = args[0]
            # alphacsc adds the loss twice in pobj per epoch, after each update of z and D
            # Divide by two the epoch number
            loss, epoch = -1 if len(pobj) == 0 else pobj[-1], len(pobj) // 2
        else:
            epoch, loss = args

        metrics = {}
        if outlier_detection_timing == "during":
            metrics = get_outliers_metric(
                info_contam["outliers_mask"].max(axis=1, keepdims=True), model, X
            )

        D_hat = model.D_hat_ if cdl_package == "wincdl" else model.D_hat
        recovery_score = evaluate_D_hat(D_true, D_hat)
        results.append(
            {
                "name": summary_name,
                **outlier_detection_method,
                **info_contam,
                **metrics,
                "recovery_score": recovery_score,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
            }
        )

    # Run the experiment
    if cdl_package == "wincdl":
        cdl = WinCDL(
            **cdl_params,
            D_init=D_init,
            outliers_kwargs=outliers_kwargs,
            callbacks=[callback_fn],
        )
        cdl.fit(X)
    elif cdl_package == "alphacsc":
        cdl_params = {
            "n_atom": D_init.shape[0],
            "n_times_atom": D_init.shape[1],
            **cdl_params,
        }
        cdl = BatchCDL(**cdl_params, D_init=D_init)
        cdl.raise_on_increase = False
        cdl.callback = callback_fn
        cdl.fit(X)
    else:
        raise ValueError(f"Unknown CDL package {cdl_package}")

    if i == 0:
        # Plot result dictionary
        fig = plot_dicts(cdl.D_hat_)
        fig.savefig(exp_dir / f"D_hat_{summary_name.replace(' ', '_')}.pdf")

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
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory to store the results",
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

    n_runs = args.n_runs
    reg = args.reg

    exp_dir = Path(args.output) / f"outlier_detection_reg_{reg}"
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
        "n_times_atom": 64,
        "n_atoms_extra": 2,  # extra atoms in the learned dictionary
        "D_init": "random",
        "window": True,
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

    # Define base CDL parameters
    # cdl_packages = ["wincdl", "alphacsc", "sporco"]
    cdl_packages = ["wincdl", "alphacsc"]
    cdl_configs = {
        "wincdl": {
            "lmbd": reg,
            "scale_lmbd": True,
            "epochs": 30,
            "max_batch": 10,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "linesearch",
            "n_iterations": 50,
            "window": True,
            "device": DEVICE,
        },
        "alphacsc": {
            "reg": reg,
            "lmbd_max": "scaled",
            "n_iter": 30,
            "solver_z": "lgcd",
            "rank1": False,
            "window": True,
            "verbose": 0,
        },
        "sporco": {},
    }

    # Define outlier detection methods
    outlier_detection_methods = [
        {"method": "none", "alpha": -1},
        # {"method": "quantile", "alpha": 0.1},
        {"method": "quantile", "alpha": 0.2},
        {"method": "iqr", "alpha": 1.5},
        # {"method": "zscore", "alpha": 1.5},
        {"method": "mad", "alpha": 3.5},
    ]
    outlier_detection_timings = ["before", "during", "never"]
    outliers_kwargs = dict(
        moving_average=None,
        union_channels=True,
        opening_window=True,
    )

    run_configs = generate_run_config_list(
        cdl_packages=cdl_packages,
        outlier_detection_methods=outlier_detection_methods,
        outlier_detection_timings=outlier_detection_timings,
        cdl_configs=cdl_configs,
        n_runs=n_runs,
        seed=seed,
    )

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            **run_config,
            outliers_kwargs=outliers_kwargs,
            simulation_params=simulation_params,
            exp_dir=exp_dir,
        )
        for run_config in run_configs
    )
    results = list(
        r for res in tqdm(results, "Running", total=len(run_configs)) for r in res
    )

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    # Plot recovery score
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    curves = (
        df_results.groupby(["name", "epoch"])["recovery_score"]
        .quantile([0.2, 0.5, 0.8])
        .unstack()
    )
    for i, name in enumerate(df_results.name.unique()):
        ax.fill_between(
            curves.loc[name].index,
            curves.loc[name, 0.2],
            curves.loc[name, 0.8],
            alpha=0.3,
            color=f"C{i}",
            label=None,
        )
        curves.loc[name, 0.5].plot(label=name, c=f"C{i}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recovery score")
    ax.legend()
    fig.savefig(exp_dir / "recovery_score.pdf")
