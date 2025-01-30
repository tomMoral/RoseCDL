from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from alphacsc import BatchCDL
from alphacsc.loss_and_gradient import compute_objective
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils.convolution import construct_X_multi
from joblib import Memory, Parallel, delayed
from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import plot_dicts
from rosecdl.utils.utils_signal import generate_experiment
from sporco.dictlrn import cbpdndl
from torch import cuda
from tqdm import tqdm

mem = Memory(location="__cache__", verbose=0)

EXP_DIR = Path("results") / "full_loss_times"
EXP_DIR.mkdir(exist_ok=True, parents=True)


@mem.cache
def run_one(
    cdl_package: str,
    cdl_params: dict[str, str or float],
    seed: int,
    i: int,
    simulation_params: dict[str, str or float],
    exp_dir: str,
):
    """Run the experiment for comparing loss and convergence time between CDL packages.

    Args:
        cdl_package (str): CDL package name: "rosecdl" or "alphacsc"
        cdl_params (dict): Parameters for the CDL algorithm
        seed (int): Random seed
        i (int): Counting index of the run
        simulation_params (dict): Parameters for data simulation
        exp_dir (str): Name of the directory to store the results
    """
    # Generate the data
    simulation_params["rng"] = seed
    X, z, D_true, D_init = generate_experiment(simulation_params)

    if i == 0:
        fig = plot_dicts(D_true)
        fig.savefig(exp_dir / "dict_true.pdf")

    # Setup the callback and timing
    results = []
    start_time = perf_counter()

    def callback_fn(model, *args):
        current_time = perf_counter() - start_time
        if len(args) == 1:
            pobj = args[0]
            loss, epoch = -1 if len(pobj) == 0 else pobj[-1], len(pobj) // 2
        else:
            epoch, loss = args

        if cdl_package == "alphacsc":
            D = model.D_hat

            z_hat, _, _ = update_z_multi(X, D, reg=cdl_configs["alphacsc"]["reg"])
            X_hat = construct_X_multi(z_hat, D=D)
            cost = compute_objective(
                X, X_hat=X_hat, z_hat=z_hat, D=D, reg=cdl_configs["alphacsc"]["reg"]
            )
        elif cdl_package == "rosecdl":
            D = model.D_hat_

            z_hat, _, _ = update_z_multi(X, D, reg=cdl_configs["rosecdl"]["lmbd"])
            X_hat = construct_X_multi(z_hat, D=D)
            cost = compute_objective(
                X,
                X_hat=X_hat,
                z_hat=z_hat,
                D=D,
                reg=cdl_configs["rosecdl"]["lmbd"],
            )

        elif cdl_package == "sporco":
            D = model.getdict().transpose(2, 1, 0)

            z_hat = model.getcoef().transpose(2, 1, 0)
            X_hat = construct_X_multi(z_hat, D=D)
            cost = compute_objective(
                X,
                X_hat=X_hat,
                z_hat=z_hat,
                D=D,
                reg=cdl_configs["sporco"]["lmbda"],
            )

        results.append(
            {
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "cost": cost,
                "time": current_time,
                "method": cdl_package,
            }
        )

    # Run the experiment
    if cdl_package == "rosecdl":
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
    elif cdl_package == "sporco":
        opt_cbpdn = cbpdndl.ConvBPDNOptionsDefaults()

        opt = cbpdndl.ConvBPDNDictLearn.Options(
            {
                "Verbose": False,
                "MaxMainIter": cdl_params["n_iter"] + 1,
                "CBPDN": opt_cbpdn,
            },
            dmethod="cns",
        )

        sporco_params = dict(
            D0=D_init.transpose(2, 1, 0).copy(),
            S=X.transpose(2, 1, 0).copy(),
            lmbda=cdl_params["lmbda"],
            opt=opt,
            dmethod="cns",
            dimN=1,
        )

        cdl = cbpdndl.ConvBPDNDictLearn(**sporco_params)
        cdl.solve()
    else:
        raise ValueError(f"Unknown CDL package: {cdl_package}")

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
        "--reg", type=float, default=0.3, help="Regularization parameter"
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

    # Simplified simulation parameters
    simulation_params = {
        "n_trials": 10,
        "n_channels": 2,
        "n_times": 5000,
        "n_atoms": 2,
        "n_times_atom": 64,
        "n_atoms_extra": 2,  # extra atoms in the learned dictionary
        "D_init": "random",
        "window": True,
        "init_d": "shapes",
        "init_d_kwargs": {"shapes": ["sin", "gaussian"]},
        "init_z": "constant",
        "init_z_kwargs": {"value": 1},
        "noise_std": 0.01,
        "rng": None,
        "sparsity": 20,
    }
    simulation_params["n_patterns_per_atom"] = simulation_params["n_channels"]

    cdl_packages = ["rosecdl", "alphacsc"]
    cdl_configs = {
        "rosecdl": {
            "lmbd": reg,
            "scale_lmbd": True,
            "epochs": 100,
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
            "n_iter": 100,
            "solver_z": "lgcd",
            "rank1": False,
            "window": True,
            "verbose": 0,
        },
        "sporco": {
            "lmbda": reg,
            "n_iter": 4,
        },
    }

    run_configs = [
        {
            "cdl_package": package,
            "cdl_params": cdl_configs[package],
            "seed": seed + i,
            "i": i,
            "exp_dir": EXP_DIR,
        }
        for package in cdl_packages
        for i in range(n_runs)
    ]

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            **run_config,
            simulation_params=simulation_params,
        )
        for run_config in run_configs
    )
    results = list(
        r for res in tqdm(results, "Running", total=len(run_configs)) for r in res
    )

    # Save results
    df = pd.DataFrame(results)
    # Add underscores to filename
    packages_str = "_".join(cdl_packages)
    df.to_csv(EXP_DIR / f"results_{packages_str}.csv", index=False)
