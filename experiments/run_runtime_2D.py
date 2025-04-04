import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
from sporco.dictlrn import cbpdndl
from tqdm import tqdm

from rosecdl.loss._ReconstructionLoss import get_lambda_max
from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat

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
    data_path: str,
    cdl_package: str,
    cdl_params: dict[str, str or float],
    seed: int,
    i: int,
) -> list:
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        data_path (str): Path to the data file.
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

    data_params = np.load(data_path)
    data = data_params["X"]
    true_dict = data_params["D"]

    rng = np.random.default_rng(seed)
    init_dict = rng.standard_normal(
        (cdl_params["n_components"],
        cdl_params["n_channels"],
        *cdl_params["kernel_size"])
    )

    lmbd_max = get_lambda_max(data, init_dict)

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

    def callback_fn(model, *args) -> None:
        global t_start, z0
        runtime = time.perf_counter() - t_start

        if cdl_package == "sporco":
            # Sporco returns the dimension of the dictionary in a different order
            model_dict = model.getdict()[:, :, 0, :, :].transpose(3, 2, 1, 0)
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

        # XXX: Calculate full loss for 2D
        # z0 = z_hat = update_z_multi(
        #     data, model_dict.astype(np.float64), lmbd, z0=z0, solver="lgcd"
        # )[0]
        # loss_true = compute_X_and_objective_multi(data, z_hat, model_dict, lmbd)
        results.append(
            {
                "name": cdl_package,
                "recovery_score": recovery_score,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                # "loss_true": loss_true,
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
        cdl.fit(data)
    elif cdl_package == "sporco":
        opt_cbpdn = cbpdndl.ConvBPDNOptionsDefaults()

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
            "lmbda": cdl_params["lmbda"],
            "opt": opt,
            "dmethod": "cns",
            "dimN": 1,
        }

        cdl = cbpdndl.ConvBPDNDictLearn(**sporco_params)
        cdl.solve()
    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

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
        "n_trials": 10,
        "n_channels": 2,
        "n_times": 1000 if args.debug else 5000,
        "n_atoms": 2,
        "n_times_atom": 64,
        "n_atoms_extra": 2,  # extra atoms in the learned dictionary
        "D_init": "random",
        "window": True,
        "contamination_params": None,
        "init_d": "shapes",
        "init_d_kwargs": {"shapes": ["sin", "gaussian", "triangle"]},
        "init_z": "constant",
        "init_z_kwargs": {"value": 1},
        "noise_std": 0.01,
        "rng": None,
        "sparsity": 20,
    }
    simulation_params["n_patterns_per_atom"] = simulation_params["n_channels"]

    # Define base CDL parameters
    cdl_packages = ["deepcdl", "rosecdl"]
    cdl_configs = {
        "rosecdl": {
            "lmbd": reg,
            "scale_lmbd": False,
            "epochs": 5 if args.debug else 30,
            "max_batch": None,
            "mini_batch_size": 10,
            "sample_window": 1000,
            "optimizer": "linesearch",
            "n_iterations": 5 if args.debug else 50,
            "window": True,
            "device": DEVICE,
        },
        "sporco": {
            "lmbda": reg,
            "n_iter": 5 if args.debug else 30,
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
    plt.show()
