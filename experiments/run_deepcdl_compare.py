"""This file contains the code to run the experiments for the outliers detection task
using the RoseCDL algorithm. It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alphacsc import BatchCDL
from alphacsc.loss_and_gradient import compute_X_and_objective_multi
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils.dictionary import get_lambda_max
from joblib import Memory, Parallel, delayed
from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat
from rosecdl.utils.utils_signal import generate_experiment
from torch import cuda
from tqdm import tqdm

mem = Memory(location="__cache__", verbose=0)


def generate_run_config_list(
    cdl_packages,
    cdl_configs,
    n_runs=1,
    seed=None,
):
    """Generate the list of configurations for the experiment.

    Args:
        cdl_packages (list): List of CDL packages to use.
        cdl_configs (dict): Dictionary of CDL configurations.
        n_runs (int): Number of runs to generate.
        seed (int): Master seed for the experiment.
    """
    # Generate a list of seeds for reproducibility
    rng = np.random.default_rng(seed)
    list_seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    run_config_list = []
    for package in cdl_packages:
        run_config_list.extend(
            {
                "cdl_package": package,
                "cdl_params": cdl_configs[package],
                "seed": s,
                "i": i,
            }
            for i, s in enumerate(list_seeds)
        )
    return run_config_list


@mem.cache
def run_one(
    simulation_params: dict[str, str or float],
    cdl_package: str,
    cdl_params: dict[str, str or float],
    seed: int,
    i: int,
):
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        simulation_params (dict): Parameters for data simulation.
        cdl_package (str): CDL package name: "rosecdl", "alphacsc" or "sporco".
        cdl_params (dict): Parameters for the CDL algorithm.
        seed (int): Random seed.
        i (int): Counting index of the run.

    """

    # Generate the data
    simulation_params["rng"] = seed
    X, z, D_true, D_init, info_contam = generate_experiment(
        simulation_params,
        return_info_contam=True,
    )
    lmbd_max = get_lambda_max(X, D_init).max()

    cdl_params = cdl_params.copy()
    if cdl_package == "alphacsc":
        cdl_params["reg"] *= lmbd_max
        lmbd = cdl_params["reg"]
    elif cdl_package in ["rosecdl", "deepcdl"]:
        cdl_params["lmbd"] *= lmbd_max
        lmbd = cdl_params["lmbd"]

    # Setup the callback
    global t_start, z0
    results, z0, t_start = [], None, time.perf_counter()

    def callback_fn(model, *args):
        global t_start, z0
        runtime = time.perf_counter() - t_start
        if len(args) == 1:
            pobj = args[0]
            # alphacsc adds the loss twice in pobj per epoch, after each update of z and D
            # Divide by two the epoch number
            loss, epoch = -1 if len(pobj) == 0 else pobj[-1], len(pobj) // 2
        else:
            epoch, loss = args

        D_hat = model.D_hat_ if cdl_package in ["rosecdl", "deepcdl"] else model.D_hat
        recovery_score = evaluate_D_hat(D_true, D_hat)
        z0 = z_hat = update_z_multi(
            X, D_hat.astype(np.float64), lmbd, z0=z0, solver="lgcd"
        )[0]
        loss_true = compute_X_and_objective_multi(X, z_hat, D_hat, lmbd)
        results.append(
            {
                "name": cdl_package,
                "recovery_score": recovery_score,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "loss_true": loss_true,
                "time": runtime,
            }
        )
        t_start = time.perf_counter()

    # Run the experiment
    if cdl_package in ["rosecdl", "deepcdl"]:
        cdl = RoseCDL(
            **cdl_params,
            D_init=D_init,
            callbacks=[callback_fn],
        )
        cdl.fit(X)
    elif cdl_package == "alphacsc":
        cdl_params = {
            "n_atoms": D_init.shape[0],
            "n_times_atom": D_init.shape[2],
            **cdl_params,
        }
        cdl = BatchCDL(**cdl_params, D_init=D_init)
        cdl.raise_on_increase = False
        cdl.callback = callback_fn
        cdl.fit(X)
    else:
        raise ValueError(f"Unknown CDL package {cdl_package}")

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

    exp_dir = Path(args.output) / f"deepcdl_comparison_{reg}"
    exp_dir.mkdir(exist_ok=True, parents=True)

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
        "contamination_params": None,
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
    cdl_packages = ["alphacsc", "deepcdl", "rosecdl"]
    cdl_configs = {
        "rosecdl": {
            "lmbd": reg,
            "scale_lmbd": False,
            "epochs": 30,
            "max_batch": None,
            "mini_batch_size": 10,
            "sample_window": 1000,
            "optimizer": "linesearch",
            "n_iterations": 50,
            "window": True,
            "device": DEVICE,
        },
        "alphacsc": {
            "reg": reg,
            "lmbd_max": "fixed",
            "n_iter": 30,
            "solver_z": "lgcd",
            "rank1": False,
            "window": True,
            "verbose": 0,
        },
        "sporco": {},
    }
    cdl_configs["deepcdl"] = cdl_configs["rosecdl"].copy()
    cdl_configs["deepcdl"]["deepcdl"] = True

    run_configs = generate_run_config_list(
        cdl_packages=cdl_packages,
        cdl_configs=cdl_configs,
        n_runs=n_runs,
        seed=seed,
    )

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(simulation_params=simulation_params, **run_config)
        for run_config in run_configs
    )
    results = list(
        r for res in tqdm(results, "Running", total=len(run_configs)) for r in res
    )

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    # Plot loss for the different methods
    curves = df_results.groupby(["name", "epoch"])[["time", "loss_true"]].median()
    _, ax = plt.subplots()
    for name in df_results.name.unique():
        c = curves.loc[name]
        c["time"] = c["time"].cumsum()
        c["loss_true"] = c["loss_true"] - df_results["loss_true"].min() + 1e1
        c.plot(x="time", y="loss_true", label=name, ax=ax)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(exp_dir / "loss_true.png")
    plt.show()
