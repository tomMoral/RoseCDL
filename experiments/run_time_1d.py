import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from alphacsc import BatchCDL
from alphacsc.loss_and_gradient import compute_X_and_objective_multi
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils.dictionary import get_lambda_max
from joblib import Memory, Parallel, delayed
from sporco.dictlrn import cbpdndl
from tqdm import tqdm

from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat
from rosecdl.utils.utils_signal import generate_experiment

mem = Memory(location="__cache__", verbose=0)

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_run_config_list(
    cdl_packages: list[str],
    cdl_configs: dict[str, dict],
    n_runs: int = 1,
    seed: int | None = None,
) -> list[dict[str, any]]:
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
) -> list:
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        simulation_params (dict): Parameters for data simulation.
        cdl_package (str): CDL package name: "rosecdl", "alphacsc" or "sporco".
        cdl_params (dict): Parameters for the CDL algorithm.
        seed (int): Random seed.
        i (int): Counting index of the run.

    """
    logger.info(
        "Running %s with seed %d (run %d)",
        cdl_package,
        seed,
        i,
    )

    # Generate the data
    simulation_params["rng"] = seed
    data, z, true_dict, init_dict, info_contam = generate_experiment(
        simulation_params,
        return_info_contam=True,
    )
    lmbd_max = get_lambda_max(data, init_dict).max()

    # Split the data into training and test sets
    n_trials = data.shape[0]
    test_data = data[: n_trials // 2]
    data = data[n_trials // 2 :]

    cdl_params = cdl_params.copy()
    if cdl_package == "alphacsc":
        cdl_params["reg"] *= lmbd_max
        lmbd = cdl_params["reg"]
    elif cdl_package in ["rosecdl", "deepcdl"]:
        cdl_params["lmbd"] *= lmbd_max
        lmbd = cdl_params["lmbd"]
    elif cdl_package == "sporco":
        cdl_params["lmbda"] *= lmbd_max
        lmbd = cdl_params["lmbda"]
    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

    # Setup the callback
    global t_start, z0
    results, t_start = [], time.perf_counter()
    z0_dict = {"train": None, "test": None}

    def callback_fn(model, *args) -> None:
        global t_start, z0
        runtime = time.perf_counter() - t_start

        if cdl_package == "sporco":
            logger.info("Sporco callback")
            # Sporco returns the dimension of the dictionary in a different order
            model_dict = model.getdict()[:, :, 0, :].transpose(2, 1, 0).copy()
            model_dict /= np.linalg.norm(model_dict, axis=(1, 2), keepdims=True)
            epoch, loss = len(results), None
        elif cdl_package == "alphacsc":
            pobj = args[0]
            # alphacsc adds the loss twice in pobj per epoch, after updating z and D
            # Divide by two the epoch number
            loss, epoch = -1 if len(pobj) == 0 else pobj[-1], len(pobj) // 2
            model_dict = model.D_hat
        else:
            epoch, loss = args
            model_dict = model.D_hat_

        recovery_score = evaluate_D_hat(true_dict, model_dict)

        # Train loss
        z_hat = update_z_multi(
            data, model_dict.astype(np.float64), lmbd, z0=z0_dict["train"], solver="lgcd"
        )[0]
        loss_true = compute_X_and_objective_multi(data, z_hat, model_dict, lmbd)
        z0_dict["train"] = z_hat

        # Test loss using the learned dictionary
        test_z_hat = update_z_multi(
            test_data, model_dict.astype(np.float64), lmbd, z0=z0_dict["test"], solver="lgcd"
        )[0]
        test_loss_true = compute_X_and_objective_multi(
            test_data, test_z_hat, model_dict, lmbd
        )
        z0_dict["test"] = test_z_hat
        results.append(
            {
                "name": cdl_package,
                "recovery_score": recovery_score,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "loss_true": loss_true,
                "test_loss_true": test_loss_true,
                "time": runtime,
            }
        )

        t_start = time.perf_counter()

    # Run the experiment
    if cdl_package in ["rosecdl", "deepcdl"]:
        if not isinstance(init_dict, torch.Tensor):
            init_dict = torch.tensor(init_dict, device=cdl_params["device"])
        cdl = RoseCDL(
            **cdl_params,
            D_init=init_dict,
            callbacks=[callback_fn],
        )
        # Setting up classical conv instead of fftconv for deepcdl
        if cdl_package == "deepcdl":
            cdl.csc.conv_algo = "classical"
            cdl.csc.set_conv_methods()
        cdl.fit(data)
    elif cdl_package == "alphacsc":
        cdl_params = {
            "n_atoms": init_dict.shape[0],
            "n_times_atom": init_dict.shape[2],
            **cdl_params,
        }
        cdl = BatchCDL(**cdl_params, D_init=init_dict)
        cdl.raise_on_increase = False
        cdl.callback = callback_fn
        cdl.fit(data)
    elif cdl_package == "sporco":
        opt_cbpdn = cbpdndl.ConvBPDNOptionsDefaults()

        opt_cbpdn["Verbose"] = False
        opt_cbpdn["AuxVarObj"] = False

        opt = cbpdndl.ConvBPDNDictLearn.Options(
            {
                "Verbose": False,
                "MaxMainIter": cdl_params["n_iter"],
                "CBPDN": opt_cbpdn,
                "Callback": callback_fn,
            },
            dmethod="cns",
        )

        sporco_params = {
            "D0": init_dict.transpose(2, 1, 0).copy(),
            "S": data.transpose(2, 1, 0).copy(),
            "lmbda": lmbd,
            "opt": opt,
            "dmethod": "cns",
            "dimN": 1,
        }

        cdl = cbpdndl.ConvBPDNDictLearn(**sporco_params)
        cdl.solve()
    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

    plot = True
    if plot:
        # plot the learned dictionary
        if cdl_package in ["rosecdl", "deepcdl"]:
            model_dict = cdl.D_hat_
        elif cdl_package == "alphacsc":
            model_dict = cdl.D_hat
        else:
            model_dict = cdl.getdict()[:, :, 0, :].transpose(2, 1, 0).copy()
        model_dict /= np.linalg.norm(model_dict, axis=(1, 2), keepdims=True)

        # Plot the learned dictionary
        fig, ax = plt.subplots(
            nrows=model_dict.shape[0], ncols=1, figsize=(10, 2 * model_dict.shape[0])
        )
        for i in range(model_dict.shape[0]):
            ax[i].plot(model_dict[i, 0, :])
        plt.tight_layout()
        plt.savefig(
            Path(f"{exp_dir}/learned_dict_{cdl_package}_{seed}_{i}.png"), dpi=300, bbox_inches="tight"
        )
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
    parser.add_argument(
        "--debug", action="store_true", help="Run the script in debug mode"
    )
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", DEVICE)

    seed = args.seed
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**32 - 1)
    logger.info("Seed: %s", seed)

    n_runs = 1 if args.debug else args.n_runs
    reg = args.reg

    exp_dir = Path(args.output) / f"runtime_comparison_{reg}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Base simulation parameters
    simulation_params = {
        "n_trials": 2*6,
        "n_channels": 1,
        "n_times": 1_000 if args.debug else 10_000,
        "n_atoms": 2,
        "n_times_atom": 128,
        "n_atoms_extra": 2,  # extra atoms in the learned dictionary
        "D_init": "random",
        "window": True,
        "contamination_params": None,
        "init_d": "shapes",
        "init_d_kwargs": {"shapes": ["sin", "gaussian"]},
        "init_z": "constant",
        "init_z_kwargs": {"value": 1},
        "noise_std": 0.01,
        "sparsity": None,
    }
    simulation_params["n_patterns_per_atom"] = simulation_params["n_channels"]
    simulation_params["sparsity"] = 20 * (simulation_params["n_times"] // 5_000)

    # Define base CDL parameters
    # cdl_packages = ["alphacsc", "deepcdl", "rosecdl", "sporco"]
    cdl_packages = ["rosecdl", "sporco"]
    cdl_configs = {
        "rosecdl": {
            "lmbd": reg,
            "scale_lmbd": False,
            "epochs": 5 if args.debug else 30,
            "max_batch": None,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "linesearch",
            "n_iterations": 5 if args.debug else 50,
            "window": False,
            "device": DEVICE,
        },
        "alphacsc": {
            "reg": reg,
            "lmbd_max": "fixed",
            "n_iter": 5 if args.debug else 30,
            "solver_z": "lgcd",
            "rank1": False,
            "window": True,
            "verbose": 1,
            "eps": 1e-6,
        },
        "sporco": {
            "lmbda": reg,
            "n_iter": 5 if args.debug else 20,
        },
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
    results = [
        r for res in tqdm(results, "Running", total=len(run_configs)) for r in res
    ]

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    # Adding the first loss evaluation of alphacsc to the other methods
    # This is done to have all the methods starting at the same time and epoch
    epoch0alphacsc = df_results[
        (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0)
    ]
    for name in df_results["name"].unique():
        if name == "alphacsc":
            continue
        epoch0alphacsc.loc[:, "name"] = name
        df_results.loc[df_results["name"] == name, "epoch"] += 1
        df_results = pd.concat([df_results, epoch0alphacsc], ignore_index=True)

    # Plotting train loss for the different methods
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["loss_true"] = median_curve["loss_true"] - df_results["loss_true"].min() + 1e1

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["loss_true"] = q02_curve["loss_true"] - df_results["loss_true"].min() + 1e1

        q8_curve = name_data[["time", "loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["loss_true"] = q8_curve["loss_true"] - df_results["loss_true"].min() + 1e1

        line, = ax.plot(median_curve["time"], median_curve["loss_true"], label=name)
        color = line.get_color()
        ax.fill_between(median_curve["time"], q02_curve["loss_true"], q8_curve["loss_true"], color=color, alpha=0.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Train Loss")
    plt.title("Speed of convergence of the different methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "loss_true.png")
    plt.show()

    # Test loss for the different methods
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "test_loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["test_loss_true"] = (
            median_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "test_loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["test_loss_true"] = (
            q02_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        q8_curve = name_data[["time", "test_loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["test_loss_true"] = (
            q8_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        (line,) = ax.plot(
            median_curve["time"], median_curve["test_loss_true"], label=name
        )
        color = line.get_color()
        ax.fill_between(
            median_curve["time"],
            q02_curve["test_loss_true"],
            q8_curve["test_loss_true"],
            color=color,
            alpha=0.2,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Loss")
    plt.title("Test performance of the different methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "test_loss_true.png")
    plt.show()
