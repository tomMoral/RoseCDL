"""This file contains the code to run the experiments for the outliers detection task
using the RoseCDL algorithm on 2D image data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from torch import cuda
from tqdm import tqdm

from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat, get_method_name, get_outliers_metric
from rosecdl.utils.utils_outliers import add_outliers_2d

mem = Memory(location="__cache__", verbose=0)
EXP_DIR = Path("results") / "outlier_detection_2d"
EXP_DIR.mkdir(exist_ok=True, parents=True)


@mem.cache
def run_one(method, outliers_kwargs, rosecdl_params, X, D_true, i, seed, exp_dir):
    # Add outliers to the clean data
    X_corrupted, outliers_mask = add_outliers_2d(
        X, contmination=0.1, patch_size=None, strength=0.6, seed=seed
    )

    # retrieve the method
    method_name = get_method_name(method)
    if method is None:
        method = {}
        this_outliers_kwargs = None
    else:
        this_outliers_kwargs = dict(**outliers_kwargs, **method)

    # Plot true dictionary if first run
    if i == 0:
        fig, ax = plt.subplots(1, D_true.shape[0], figsize=(10, 5))
        for j, atom in enumerate(D_true):
            ax[j].imshow(atom.squeeze(), cmap="gray")
            ax[j].axis("off")
        plt.savefig(exp_dir / "dict_true.pdf")

    # Setup the callback
    results = []

    def callback_fn(model, epoch, loss):
        if hasattr(model.loss_fn, "method"):
            metrics = get_outliers_metric(
                outliers_mask,
                model,
                X_corrupted,  # Use corrupted data and true mask
            )
        else:
            metrics = {}
        recovery_score = evaluate_D_hat(D_true[:, None], model.D_hat_)

        results.append(
            {
                "name": method_name,
                **method,
                **metrics,
                "recovery_score": recovery_score,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
            }
        )

    # Run the experiment with corrupted data
    rosecdl = RoseCDL(
        **rosecdl_params,
        outliers_kwargs=this_outliers_kwargs,
        callbacks=[callback_fn],
        random_state=seed,
    )
    rosecdl.fit(X_corrupted)

    # Plot learned dictionary if first run
    if i == 0:
        fig, ax = plt.subplots(1, rosecdl.D_hat_.shape[0], figsize=(10, 5))
        for j, atom in enumerate(rosecdl.D_hat_):
            ax[j].imshow(atom.squeeze(), cmap="gray")
            ax[j].axis("off")
        plt.savefig(exp_dir / f"D_hat_{method_name}.pdf")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run outlier detection comparison on 2D image data"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to NPZ file containing the data (X and d arrays)",
    )
    parser.add_argument("--n-jobs", "-j", type=int, default=1)
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--n-runs", "-n", type=int, default=20)
    parser.add_argument("--reg", type=float, default=0.8)
    args = parser.parse_args()

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    seed = args.seed if args.seed is not None else np.random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")
    rng = np.random.default_rng(seed)

    exp_dir = EXP_DIR / f"outlier_detection_2d_reg_{args.reg}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Load the data
    data = np.load(args.data_path)
    X = data.get("X")
    D_true = data.get("D")

    if X.ndim == 2:
        X = X[None, None]

    # Define base detection parameters
    outliers_kwargs = dict(
        moving_average=None,
        union_channels=False,
        opening_window=True,
    )

    rosecdl_params = {
        "n_components": 6,
        "kernel_size": (35, 35),
        "n_channels": 1,
        "lmbd": args.reg,
        "scale_lmbd": True,
        "epochs": 30,
        "max_batch": 1,
        "mini_batch_size": 1,
        "sample_window": 500,
        "optimizer": "linesearch",
        "n_iterations": 50,
        "window": True,
        "device": DEVICE,
    }

    # Define list of methods to test
    alpha_true = 0.1
    list_methods = [
        None,
        {"method": "quantile", "alpha": alpha_true},
        {"method": "iqr", "alpha": 1.5},
        {"method": "zscore", "alpha": 2},
        {"method": "mad", "alpha": 3.5},
    ]

    list_seeds = rng.integers(0, 2**32 - 1, args.n_runs)
    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            this_method, outliers_kwargs, rosecdl_params, X, D_true, i, seed, exp_dir
        )
        for i, seed in enumerate(list_seeds)
        for this_method in list_methods
    )
    results = list(
        r
        for res in tqdm(results, "Running", total=args.n_runs * len(list_methods))
        for r in res
    )

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    # Plot recovery scores
    curves = df_results.groupby(["name", "epoch"])["recovery_score"].mean()
    plt.figure(figsize=(10, 6))
    for name in df_results.name.unique():
        curves.loc[name].plot(label=name)
    plt.legend()
    plt.title("Recovery Score vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Recovery Score")
    plt.savefig(exp_dir / "recovery_score.pdf")
