"""This file contains the code to run the experiments for the outliers detection task
using the RoseCDL algorithm. It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat, get_outliers_metric
from rosecdl.utils.utils_outlier_comparison import remove_outliers_before_cdl
from rosecdl.utils.utils_outliers import add_outliers_2d
from torch import cuda
from tqdm import tqdm

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
            if package != "rosecdl" and timing == "during":
                # Only rosecdl can run the outlier detection during the CDL
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
    data_path,
    exp_dir: str,
):
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        cdl_package (str): CDL package name: "rosecdl", "alphacsc" or "sporco".
        outlier_detection_method (str): Outlier detection method.
            A dictionary is expected with two keys:
                - "name": name of the method. Can be one of "quantile", "iqr",
                    "zscore", "mad" or "none".
                - "alpha" (float): Parameter of the method.
        outlier_detection_timing (str): When to run the outlier detection
            relatively to the CDL: "before", "during" or "never".
        outliers_kwargs (dict): Additional parameters for outlier detection.
        cdl_params (dict): Parameters for the CDL algorithm.
        seed (int): Random seed.
        i (int): Counting index of the run.
        exp_dir (str): Name of the directory to store the results

    """
    print(f"Running {cdl_package} with {outlier_detection_timing} ({seed})")

    if cdl_package == "sporco":
        return []

    data_params = np.load(data_path)

    X = data_params["X"]
    D_true = data_params["D"]

    X = X[None, None, ...]
    assert X.ndim == 4  # (n_trials, n_channels, height, width)

    D_true = np.expand_dims(D_true, axis=1)
    assert D_true.ndim == 4  # (n_atoms, n_channels, height, width)

    # Add outliers to the data
    X_outliers, outliers_mask = add_outliers_2d(
        X, contamination=0.1, patch_size=None, strength=5, seed=seed
    )

    # X = X_outliers[:, :, :900, :900]
    # outliers_mask = outliers_mask[:, :, :900, :900]

    # if outliers_mask.sum() == 0:  # At least one outlier
    #     X_outliers, outliers_mask = add_outliers_2d(
    #         X, contamination=0.1, patch_size=None, strength=5, seed=seed
    #     )
    #     X = X_outliers[:, :, :900, :900]
    #     outliers_mask = outliers_mask[:, :, :900, :900]

    X = X_outliers

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    info_contam = {"outliers_mask": outliers_mask}

    if i == 0:
        # Plot true dictionary
        fig, axes = plt.subplots(1, D_true.shape[0], figsize=(12, 10))
        for i, ax in enumerate(axes):
            ax.imshow(D_true[i, 0], cmap="gray")
            ax.axis("off")
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
    # XXX: Get z shape otherwise :
    # zshape = (X.shape[0], X.shape[1], X.shape[-2]-kernel_size[0]+1, X.shape[-1]-kernel_size[1]+1)

    zshape = (
        X.shape[0],
        X.shape[1],
        X.shape[-2] - cdl_configs["rosecdl"]["kernel_size"][0] + 1,
        X.shape[-1] - cdl_configs["rosecdl"]["kernel_size"][1] + 1,
    )

    if outlier_detection_timing == "before":
        X = remove_outliers_before_cdl(
            data=X,
            activation_vector_shape=zshape,
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
            metrics = get_outliers_metric(info_contam["outliers_mask"], model, X)
            # metrics = get_outliers_metric(
            #     info_contam["outliers_mask"].max(axis=1, keepdims=True), model, X
            # )

        D_hat = model.D_hat_ if cdl_package == "rosecdl" else model.D_hat
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
    if cdl_package == "rosecdl":
        D_init = np.random.randn(
            cdl_configs["rosecdl"]["n_components"],
            cdl_configs["rosecdl"]["n_channels"],
            *cdl_configs["rosecdl"]["kernel_size"],
        )

        cdl = RoseCDL(
            **cdl_params,
            D_init=D_init,
            outliers_kwargs=outliers_kwargs,
            callbacks=[callback_fn],
        )
        cdl.fit(X)
    else:
        raise ValueError(f"Unknown CDL package {cdl_package}")

    if i == 0:
        # Plot result dictionary
        fig, axes = plt.subplots(1, cdl.D_hat_.shape[0], figsize=(12, 10))
        for i, ax in enumerate(axes):
            ax.imshow(cdl.D_hat_[i, 0], cmap="gray")
            ax.axis("off")
        fig.savefig(exp_dir / f"D_hat_{summary_name.replace(' ', '_')}.pdf")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a comparison of the different methods for dictionary recovery in 2D"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to NPZ file containing the data (X and d arrays)",
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

    exp_dir = Path(args.output) / f"outlier_detection_2d_reg_{reg}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Base contamination parameters
    contamination_params = {}

    # Define base CDL parameters
    # cdl_packages = ["rosecdl", "sporco"]
    cdl_packages = ["rosecdl"]
    cdl_configs = {
        "rosecdl": {
            "kernel_size": (35, 30),
            "n_channels": 1,
            "n_components": 6,
            "lmbd": reg,
            "scale_lmbd": False,
            "epochs": 30,
            "max_batch": 20,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "adam",
            "n_iterations": 60,
            "window": True,
            "device": DEVICE,
            "positive_D": True,
        },
        "sporco": {},
    }

    # Define outlier detection methods
    outlier_detection_methods = [
        {"method": "none", "alpha": -1},
        # {"method": "quantile", "alpha": 0.1},
        # {"method": "quantile", "alpha": 0.2},
        # {"method": "iqr", "alpha": 1.5},
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
            data_path=args.data_path,
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
