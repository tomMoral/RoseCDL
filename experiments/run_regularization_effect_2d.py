"""This script analyzes the effect of regularization parameter on dictionary learning
for letter data. It saves results to a CSV file for further analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib import Memory

from wincdl.datasets import create_conv_dataloader
from wincdl.utils.utils_exp import evaluate_D_hat, make_size
from wincdl.wincdl import WinCDL

mem = Memory(location="__cache__", verbose=0)
EXP_DIR = Path("results")
EXP_DIR.mkdir(exist_ok=True, parents=True)


@mem.cache
def run_one(reg, run_idx, model_params, data_params, seed):
    """Run a single experiment with given parameters."""
    # rng = np.random.default_rng(seed + run_idx)

    # Create dataloader
    dataloader = create_conv_dataloader(
        data=data_params["X"].reshape(1, 1, *data_params["X"].shape),
        sample_window=model_params["sample_window"],
        dtype=model_params["dtype"],
        device=model_params["device"],
    )

    # Setup callback
    results = []

    def callback_recovery(model, epoch, loss):
        true_D = make_size(data_params["true_D"], model.D_hat_.shape)
        recovery_score = evaluate_D_hat(
            model.D_hat_,
            true_D,
        )
        results.append(
            {
                "run_idx": run_idx,
                "epoch": epoch,
                "loss": loss,
                "recovery_score": recovery_score,
                "reg": reg,
            }
        )

    # Setup and train model
    model = WinCDL(
        **model_params,
        lmbd=reg,
        callbacks=[callback_recovery],
    )
    model.fit(dataloader)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze regularization effect on dictionary learning"
    )
    parser.add_argument(
        "--n-jobs", "-j", type=int, default=1, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Master seed for reproducible job"
    )
    parser.add_argument(
        "--n-runs",
        "-n",
        type=int,
        default=5,
        help="Number of repetitions for the experiment",
    )
    args = parser.parse_args()

    # Setup device and seed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")

    # Load data
    data = np.load("text_4_5000_ICML.npz")
    data_params = {"X": data.get("X"), "true_D": data.get("D")}

    # Model parameters
    model_params = {
        "n_components": 6,
        "kernel_size": (35, 35),
        "n_channels": 1,
        "lmbd": 0.1,
        "n_iterations": 30,
        "epochs": 100,
        "sample_window": 500,
        "mini_batch_size": 10,
        "device": DEVICE,
        "dtype": torch.float32,
    }

    # Experiment parameters
    reg_values = [0.01, 0.1] #, 0.2, 0.3, 0.4, 0.5, 0.6]
    exp_dir = EXP_DIR / "reg_effect"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Run experiments
    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(reg, run_idx, model_params, data_params, seed)
        for reg in reg_values
        for run_idx in range(args.n_runs)
    )

    # Flatten results and save
    results = [r for res in tqdm(results, "Processing results") for r in res]
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "reg_effect.csv", index=False)

    print("\nExperiment completed! Results saved to reg_effect.csv")
