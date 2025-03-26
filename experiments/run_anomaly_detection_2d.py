import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch import cuda
from tqdm import tqdm

from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import get_outliers_metric
from rosecdl.utils.utils_outlier_comparison import remove_outliers_before_cdl
from rosecdl.utils.utils_outliers import add_outliers_2d

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
    outlier_detection_methods: list[dict[str, float | str]],
    outlier_detection_timings: list[str],
    cdl_configs: dict[str, dict],
    n_runs: int = 1,
    seed: int | None = None,
) -> list[dict[str, any]]:
    """Generate the list of configurations for the experiment.

    Parameters
    ----------
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


def validate_dimensions(tensor: np.ndarray, expected_dim: int, name: str) -> None:
    """Validate tensor dimensions.

    Args:
        tensor: Input tensor to validate
        expected_dim: Expected number of dimensions
        name: Name of the tensor for error message

    """
    if tensor.ndim != expected_dim:
        msg = (
            f"Expected {name} to have {expected_dim} dimensions, "
            f"but got {tensor.ndim} dimensions"
        )
        raise ValueError(msg)


@mem.cache
def run_one(
    cdl_package: str,
    cdl_params: dict[str, str or float],
    outlier_detection_method: dict[str, str or float],
    outlier_detection_timing: str,
    seed: int,
    i: int,
    outliers_kwargs: dict[str, str or float],
    data_path: str,
    exp_dir: str,
) -> list:
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
    logger.info(
        "Running %s with %s (seed=%d)", cdl_package, outlier_detection_timing, seed
    )

    if cdl_package == "sporco":
        logger.warning("Sporco package not implemented, skipping")
        return []

    data_params = np.load(data_path)

    X = data_params["X"]
    D_true = data_params["D"]

    X = X[None, None, ...]
    validate_dimensions(X, 4, "Input data X")  # (n_trials, n_channels, height, width)

    D_true = np.expand_dims(D_true, axis=1)
    validate_dimensions(
        D_true, 4, "Dictionary D"
    )  # (n_atoms, n_channels, height, width)

    # Add outliers to the data
    X_outliers, outliers_mask = add_outliers_2d(
        X,
        contamination=0.2,
        patch_size=(80, 80),
        strength=3,
        seed=seed,
        noise=0.1,
        clip=True,
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
        for idx, ax in enumerate(axes):
            ax.imshow(D_true[idx, 0], cmap="gray")
            ax.axis("off")
        fig.savefig(exp_dir / "dict_true.pdf")

    # Process the outlier method's parameters and compute the summary name
    summary_name = "{method} (alpha={alpha:.02f})".format(
        **outlier_detection_method,
    )
    outliers_kwargs = {**outliers_kwargs, **outlier_detection_method}

    summary_name = f"[{cdl_package}] {summary_name}"
    if outlier_detection_timing != "never":
        summary_name += f" ({outlier_detection_timing})"

    # Perform outlier detection on the data before CDL
    zshape = (
        X.shape[0],
        X.shape[1],
        X.shape[-2] - cdl_configs["rosecdl"]["kernel_size"][0] + 1,
        X.shape[-1] - cdl_configs["rosecdl"]["kernel_size"][1] + 1,
    )

    if outlier_detection_timing == "before":
        X, mask_before = remove_outliers_before_cdl(
            data=X,
            activation_vector_shape=zshape,
            **outliers_kwargs,
            return_outliers_mask=True,
        )
        outliers_kwargs = None
        info_contam["mask_before"] = mask_before

    # Setup the callback
    results = []

    def callback_fn(model: any, *args: any) -> None:
        if len(args) == 1:
            pobj = args[0]
            # alphacsc adds the loss twice in pobj per epoch, after each update of z and D
            # Divide by two the epoch number
            loss, epoch = -1 if len(pobj) == 0 else pobj[-1], len(pobj) // 2
        else:
            epoch, loss = args

        metrics = {}
        true_outliers_mask = info_contam["outliers_mask"]
        dice_score_epsilon = 1e-7

        if outlier_detection_timing == "during":
            metrics = get_outliers_metric(
                true_outliers_mask, model, X, dice_score_epsilon, crop=True
            )
        elif outlier_detection_timing == "before":
            kheight, kwidth = cdl_params["kernel_size"]
            true_outliers_mask = true_outliers_mask[:, :, : -kheight + 1, : -kwidth + 1]
            mask_before = info_contam["mask_before"][
                :, :, : -kheight + 1, : -kwidth + 1
            ]

            if isinstance(true_outliers_mask, torch.Tensor):
                true_outliers_mask = true_outliers_mask.cpu().numpy()

            # Ensure masks have the same dtype
            if (
                true_outliers_mask.dtype != mask_before.dtype
                and mask_before.dtype == bool
            ):
                mask_before = mask_before.astype(np.int32)

            accuracy = np.mean(mask_before == true_outliers_mask)
            precision = precision_score(
                true_outliers_mask.flatten(), mask_before.flatten()
            )
            recall = recall_score(true_outliers_mask.flatten(), mask_before.flatten())
            f1 = f1_score(true_outliers_mask.flatten(), mask_before.flatten())
            dice = 2 * (precision * recall) / (precision + recall + dice_score_epsilon)
            jaccard = jaccard_score(true_outliers_mask.flatten(), mask_before.flatten())

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "dice": dice,
                "jaccard": jaccard,
                "percentage": np.mean(mask_before),
            }

        results.append(
            {
                "name": summary_name,
                **outlier_detection_method,
                **info_contam,
                **metrics,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
            },
        )

    # Run the experiment
    if cdl_package == "rosecdl":
        torch.manual_seed(seed)
        d_init = torch.randn(
            cdl_configs["rosecdl"]["n_components"],
            cdl_configs["rosecdl"]["n_channels"],
            *cdl_configs["rosecdl"]["kernel_size"],
        )

        cdl = RoseCDL(
            **cdl_params,
            D_init=d_init,
            outliers_kwargs=outliers_kwargs,
            callbacks=[callback_fn],
        )
        cdl.fit(X)
    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

    if i == 0:
        # Plot result dictionary
        fig, axes = plt.subplots(1, cdl.D_hat_.shape[0], figsize=(12, 10))
        for idx, ax in enumerate(axes):
            ax.imshow(cdl.D_hat_[idx, 0], cmap="gray")
            ax.axis("off")
        fig.savefig(exp_dir / f"D_hat_{summary_name.replace(' ', '_')}.pdf")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run a comparison of the different methods for anomaly detection in 2D"
        ),
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
    parser.add_argument(
        "--window", action="store_true", help="Use windowing in the CDL algorithm"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the script in debug mode"
    )
    args = parser.parse_args()

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    logger.info("Using device: %s", DEVICE)

    seed = args.seed
    if seed is None:
        rng = np.random.default_rng()
        seed = rng.integers(0, 2**32 - 1)
    logger.info("Using seed: %s", seed)

    n_runs = 1 if args.debug else args.n_runs
    reg = args.reg

    window_suffix = "window" if args.window else "no_window"
    exp_dir = Path(args.output) / f"anomaly_detection_2d_reg_{reg}_{window_suffix}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Define base CDL parameters
    cdl_packages = ["rosecdl"]
    cdl_configs = {
        "rosecdl": {
            "kernel_size": (35, 30),
            "n_channels": 1,
            "n_components": 6,
            "lmbd": reg,
            "scale_lmbd": False,
            "epochs": 10 if args.debug else 30,
            "max_batch": 20,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "adam",
            "n_iterations": 30 if args.debug else 50,
            "window": args.window,
            "device": DEVICE,
            "positive_D": True,
        },
        "sporco": {},
    }

    # Define outlier detection methods
    outlier_detection_methods = (
        [
            {"method": "mad", "alpha": 3.5},
        ]
        if args.debug
        else [
            {"method": "quantile", "alpha": 0.1},
            {"method": "quantile", "alpha": 0.2},
            {"method": "iqr", "alpha": 1.5},
            {"method": "zscore", "alpha": 1.5},
            {"method": "mad", "alpha": 3.5},
        ]
    )
    outlier_detection_timings = ["during"] if args.debug else ["before", "during"]
    outliers_kwargs = {
        "moving_average": None,
        "union_channels": True,
        "opening_window": True,
    }

    run_configs = generate_run_config_list(
        cdl_packages=cdl_packages,
        outlier_detection_methods=outlier_detection_methods,
        outlier_detection_timings=outlier_detection_timings,
        cdl_configs=cdl_configs,
        n_runs=n_runs,
        seed=seed,
    )

    # Save configuration parameters as JSON for reproducibility
    import json

    config_to_save = {
        "cdl_configs": cdl_configs,
        "outlier_detection_methods": outlier_detection_methods,
        "outlier_detection_timings": outlier_detection_timings,
        "outliers_kwargs": outliers_kwargs,
        "n_runs": n_runs,
        "seed": seed,
        "args": vars(args),
    }
    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(config_to_save, f, indent=4, default=str)

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            **run_config,
            outliers_kwargs=outliers_kwargs,
            data_path=args.data_path,
            exp_dir=exp_dir,
        )
        for run_config in run_configs
    )
    results = [
        r for res in tqdm(results, "Running", total=len(run_configs)) for r in res
    ]

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    fig_box, ax_box = plt.subplots(figsize=(8, 6))

    last_epoch = df_results["epoch"].max()
    last_epoch_data = df_results[df_results["epoch"] == last_epoch]

    box_plot = ax_box.boxplot(
        [group["f1"].to_numpy() for name, group in last_epoch_data.groupby("name")],
        tick_labels=[name for name, _ in last_epoch_data.groupby("name")],
        patch_artist=True,
    )

    for i, patch in enumerate(box_plot["boxes"]):
        patch.set_facecolor(f"C{i}")
        patch.set_alpha(0.6)

    ax_box.set_ylabel("F1 Score (Last Epoch)")
    ax_box.set_title("F1 Score for different anomaly detection methods")
    ax_box.tick_params(axis="x", rotation=90)
    ax_box.grid(linestyle="--", alpha=0.7)
    ax_box.set_ylim(0, 1)

    # Save the box plot figure
    plt.tight_layout()
    fig_box.savefig(exp_dir / "anomaly_detection_boxplot.pdf")
