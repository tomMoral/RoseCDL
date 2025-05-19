import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
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
    reg_list: list[float] | np.ndarray[float],
    cdl_configs: dict[str, dict],
    n_runs: int = 1,
    seed: int | None = None,
) -> list[dict[str, any]]:
    """Generate the list of configurations for the experiment.

    Args:
        reg_list (list): List of regularization parameters.
        cdl_configs (dict): Dictionary of CDL configurations.
        n_runs (int): Number of runs to generate.
        seed (int): Master seed for the experiment.

    """
    # Generate a list of seeds for reproducibility
    rng = np.random.default_rng(seed)
    list_seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    run_config_list = []
    for reg in reg_list:
        run_config_list.extend(
            {
                "cdl_params": cdl_configs["rosecdl"],
                "reg": reg,
                "seed": s,
                "master_seed": seed,
                "i": i,
            }
            for i, s in enumerate(list_seeds)
        )
    return run_config_list


@mem.cache
def run_one(
    simulation_params: dict[str, str | float],
    cdl_params: dict[str, str | float],
    seed: int,
    master_seed: int,
    reg: float,
    i: int,
) -> list:
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        simulation_params (dict): Parameters for data simulation.
        cdl_params (dict): Parameters for the CDL algorithm.
        seed (int): Random seed.
        master_seed (int): Master seed for reproducibility.
        reg (float): Regularization.
        i (int): Counting index of the run.

    """
    logger.info(
        "Running with regularization value %f (run %d)",
        reg,
        i,
    )

    cdl_params["lmbd"] = reg

    # Generate the data
    simulation_params["rng"] = seed
    data, z, true_dict, init_dict, info_contam = generate_experiment(
        simulation_params,
        return_info_contam=True,
    )

    results = []

    # Setup the callback
    def callback_fn(model, epoch, loss) -> None:
        model_dict = model.D_hat_

        recovery_score = evaluate_D_hat(model_dict, true_dict)

        results.append(
            {
            "recovery_score": recovery_score,
            "seed": seed,
            "master_seed": master_seed,
            "epoch": epoch,
            "loss": loss,
            "reg": reg,
            "scaled_reg": model.csc.lmbd,
            "model_dict": model_dict.tolist(),
            "true_dict": true_dict.tolist(),
            }
        )

    init_dict = torch.tensor(init_dict)
    cdl = RoseCDL(
        **cdl_params,
        D_init=init_dict,
        callbacks=[callback_fn],
    )
    cdl.fit(data)

    plot = True
    if plot:
        model_dict = cdl.D_hat_
        # Plot the learned dictionary
        fig, ax = plt.subplots(
            nrows=model_dict.shape[0], ncols=1, figsize=(10, 2 * model_dict.shape[0])
        )
        for idx in range(model_dict.shape[0]):
            ax[idx].plot(model_dict[idx, 0, :])
        plt.tight_layout()
        plt.savefig(
            Path(f"{exp_dir}/learned_dict_reg_{reg:.2f}_{seed}_{idx}.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(
            nrows=true_dict.shape[0], ncols=1, figsize=(10, 2 * true_dict.shape[0])
        )
        for idx in range(true_dict.shape[0]):
            ax[idx].plot(true_dict[idx, 0, :])
        plt.tight_layout()
        plt.savefig(
            Path(f"{exp_dir}/true_dict_reg_{reg:.2f}_{seed}_{idx}.png"),
            dpi=300,
            bbox_inches="tight",
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
    parser.add_argument(
        "--solver",
        type=str,
        nargs="+",  # Allow multiple solver names
        help="Filter by specific solver names",
        default=None,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="cuda",
        help="GPU to use for the experiment (default: cuda)",
    )
    args = parser.parse_args()

    DEVICE = args.gpu if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", DEVICE)

    master_seed = args.seed
    if master_seed is None:
        master_seed = np.random.default_rng().integers(0, 2**32 - 1)
    logger.info("Seed: %s", master_seed)

    n_runs = 1 if args.debug else args.n_runs
    reg = args.reg

    exp_dir = Path(args.output) / f"regularization_effect_{master_seed}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Base simulation parameters
    simulation_params = {
        "n_trials": 2 * 10,
        "n_channels": 1,
        "n_times": 5_000 if args.debug else 30_000,
        "n_atoms": 2,
        "n_times_atom": 128,
        "n_atoms_extra": 1,  # extra atoms in the learned dictionary
        "D_init": "random",
        "window": True,
        "contamination_params": None,
        "init_d": "shapes",
        "init_d_kwargs": {"shapes": ["sin", "gaussian"]},
        "init_z": "constant",
        "init_z_kwargs": {"value": 1},
        "noise_std": 0.00,
        "sparsity": None,
    }
    simulation_params["n_patterns_per_atom"] = simulation_params["n_channels"]
    simulation_params["sparsity"] = 10 * (simulation_params["n_times"] // 5_000)

    cdl_configs = {
        "rosecdl": {
            "lmbd": reg,
            "scale_lmbd": True,
            "epochs": 5 if args.debug else 30,
            "max_batch": None,
            "mini_batch_size": 250,
            "sample_window": 10_000,
            "optimizer": "linesearch",
            "n_iterations": 5 if args.debug else 50,
            "window": False,
            "positive_D": False,
            "device": DEVICE,
        },
    }

    reg_list = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
    run_configs = generate_run_config_list(
        reg_list=reg_list,
        cdl_configs=cdl_configs,
        n_runs=n_runs,
        seed=master_seed,
    )

    # Logging using cdl_package and seed
    logger.info("Run configurations:")
    logger.info("REGS : %s", reg_list)

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(simulation_params=simulation_params, **run_config)
        for run_config in run_configs
    )
    results = [
        r for res in tqdm(results, "Running", total=len(run_configs)) for r in res
    ]

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "results.csv", index=False)
    logger.info("Results saved to %s", exp_dir / "results.csv")
    df_results = df_results[df_results["recovery_score"] < 1]
    # Group by regularization and epoch, then compute quantiles over seeds
    grouped = df_results.groupby(["reg", "epoch"])["recovery_score"]
    median = df_results.pivot_table(
        index="epoch", columns="reg", values="recovery_score", aggfunc="median"
    )
    q20 = df_results.pivot_table(
        index="epoch",
        columns="reg",
        values="recovery_score",
        aggfunc=lambda x: x.quantile(0.2),
    )
    q80 = df_results.pivot_table(
        index="epoch",
        columns="reg",
        values="recovery_score",
        aggfunc=lambda x: x.quantile(0.8),
    )

    plt.figure(figsize=(10, 6))
    regs_sorted = sorted(df_results["reg"].unique())
    cmap = plt.get_cmap("viridis", len(regs_sorted))
    palette_regs = [cmap(i) for i in range(len(regs_sorted))]
    reg_color_map = {
        reg: palette_regs[::-1][i] for i, reg in enumerate(regs_sorted)
    }
    for reg in regs_sorted:
        color = reg_color_map[reg]
        plt.plot(
            median.index, median[reg], marker="o", label=f"reg={reg:.2f}", color=color
        )
        plt.fill_between(median.index, q20[reg], q80[reg], alpha=0.2, color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Recovery Score")
    plt.title("Recovery Score Evolution over Epochs for Different Regularizations")
    plt.legend(title="Regularization")
    plt.tight_layout()
    plt.savefig(exp_dir / "recovery_vs_reg_epochs.png", dpi=300)
    plt.show()
