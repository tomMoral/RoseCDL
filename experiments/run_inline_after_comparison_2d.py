import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch import cuda
from torch.nn import MSELoss
from tqdm import tqdm

from rosecdl.rosecdl import RoseCDL

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
        cdl_packages : list[str]
            List of CDL packages to use.
        outlier_detection_methods : list[dict[str, float | str]]
            List of outlier detection methods.
        outlier_detection_timings : list[str]
            List of outlier detection timings.
        cdl_configs : dict[str, dict]
            Dictionary of CDL configurations.
        n_runs : int, optional
            Number of runs to generate, by default 1.
        seed : int | None, optional
            Master seed for the experiment, by default None.

    Returns
    -------
        list[dict[str, any]]
            List of configurations for the experiment.

    """
    # Generate a list of seeds for reproducibility
    rng = np.random.default_rng(seed)
    list_seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    run_config_list = []
    for package in cdl_packages:
        for timing in outlier_detection_timings:
            for method in outlier_detection_methods:
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
    cdl_params: dict[str, str | float],
    outlier_detection_method: dict[str, str | float],
    outlier_detection_timing: str,
    seed: int,
    i: int,
    outliers_kwargs: dict[str, str | float],
    data_path: str,
    exp_dir: str,
) -> list:
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        cdl_package (str): CDL package name: "rosecdl".
        outlier_detection_method (str): Outlier detection method.
            A dictionary is expected with two keys:
                - "name": name of the method. Can be one of "quantile", "iqr",
                    "zscore", "mad" or "none".
                - "alpha" (float): Parameter of the method.
                - "opening_window" (tuple): Size of the opening window.
        outlier_detection_timing (str): When to run the outlier detection
            relatively to the CDL: "after", "during".
        outliers_kwargs (dict): Additional parameters for outlier detection.
        cdl_params (dict): Parameters for the CDL algorithm.
        seed (int): Random seed.
        i (int): Counting index of the run.
        data_path (str): Path to the NPZ file containing the data.
        exp_dir (str): Name of the directory to store the results

    """
    logger.info(
        "Starting run_one: cdl_package=%s, outlier_detection_timing=%s, seed=%d, i=%d",
        cdl_package,
        outlier_detection_timing,
        seed,
        i,
    )
    logger.info("Outlier detection method: %s", outlier_detection_method)
    logger.info("CDL params: %s", cdl_params)
    logger.info("Outliers kwargs: %s", outliers_kwargs)
    logger.info("Experiment directory: %s", exp_dir)

    logger.info(
        "Running %s with %s (seed=%d)", cdl_package, outlier_detection_timing, seed
    )

    data_params = np.load(data_path)

    data = data_params["X"]
    true_dict = data_params["D"]
    true_mask = data_params["mask"]

    data = data[None, None, ...]
    true_mask = true_mask[None, None, ...]
    true_dict = np.expand_dims(true_dict, axis=1)

    validate_dimensions(data, 4, "Input data X")
    validate_dimensions(true_mask, 4, "True mask")
    validate_dimensions(true_dict, 4, "Dictionary D")

    logger.info("Data loaded successfully.")
    logger.info("True dictionary shape: %s", true_dict.shape)

    data = data[:, :, :1500, :1500]
    true_mask = true_mask[:, :, :1500, :1500]
    data = torch.tensor(data, device=cdl_params["device"])

    if i == 0:
        # Plot true dictionary
        fig, axes = plt.subplots(1, true_dict.shape[0], figsize=(12, 10))
        for idx, ax in enumerate(axes):
            ax.imshow(true_dict[idx, 0], cmap="gray")
            ax.axis("off")
        logger.info("Saving true dictionary plot to %s", exp_dir / "dict_true.pdf")
        fig.savefig(exp_dir / "dict_true.pdf")
        plt.close()

        # Plot true mask
        plt.imshow(true_mask[0, 0], cmap="gray")
        plt.axis("off")
        logger.info("Saving true mask plot to %s", exp_dir / "mask_true.pdf")
        plt.savefig(exp_dir / "mask_true.pdf")
        plt.close()

        logger.info("Number of anomalies in the true mask: %d", np.sum(true_mask))
        logger.info(
            "Percentage of anomalies in the true mask: %.2f%%", np.mean(true_mask)
        )

    # Process the outlier method's parameters and compute the summary name
    logger.info("Processing outlier detection method: %s", outlier_detection_method)
    summary_name = "{method} (alpha={alpha:.02f})".format(**outlier_detection_method)
    outliers_kwargs = {**outliers_kwargs, **outlier_detection_method}
    logger.info("Summary name for results: %s", summary_name)
    logger.info("Updated outliers_kwargs: %s", outliers_kwargs)

    torch.manual_seed(seed)
    logger.info("Initializing RoseCDL model with seed %d.", seed)
    d_init = torch.randn(
        cdl_configs["rosecdl"]["n_components"],
        cdl_configs["rosecdl"]["n_channels"],
        *cdl_configs["rosecdl"]["kernel_size"],
    )

    evaluation_model = RoseCDL(**cdl_params, D_init=d_init, outliers_kwargs=None)

    results = []

    def callback_fn(model: any, epoch: int, loss: float) -> None:
        nonlocal true_mask, evaluation_model
        metrics = {}
        dice_score_epsilon = 1e-7

        model_dict = torch.tensor(model.D_hat_, device=cdl_params["device"])
        kernel_size = model.csc.kernel_size

        if outlier_detection_timing == "during":
            xh, zh = model.csc(data)
            callback_mask = model.loss_fn.get_outliers_mask(
                xh,
                zh,
                data,
                opening=outlier_detection_method["opening_window"],
                crop=True,
            )
        elif outlier_detection_timing == "after":
            # We reconstruct the data. We threshold on the reconstruction error
            X_hat, z_hat = evaluation_model.csc(data, D=model_dict)
            err = MSELoss(reduction="none")(X_hat, data).cpu().numpy()

            err = err[
                :, :, kernel_size[0] : -kernel_size[0], kernel_size[1] : -kernel_size[1]
            ]

            # Compute the threshold on the error according to the method manually
            if outlier_detection_method["method"] == "quantile":
                threshold = np.quantile(err, outlier_detection_method["alpha"])
            elif outlier_detection_method["method"] == "zscore":
                threshold = np.mean(err) + outlier_detection_method["alpha"] * np.std(
                    err
                )
            elif outlier_detection_method["method"] == "mad":
                median = np.median(err)
                mad = np.median(np.abs(err - median))
                threshold = median + outlier_detection_method["alpha"] * mad
            elif outlier_detection_method["method"] == "iqr":
                q1 = np.percentile(err, 25)
                q3 = np.percentile(err, 75)
                iqr = q3 - q1
                threshold = q3 + outlier_detection_method["alpha"] * iqr
            else:
                msg = f"OD method {outlier_detection_method['method']} not supported"
                raise ValueError(msg)
            # Compute the outliers mask
            callback_mask = err > threshold

        this_true_mask = true_mask
        this_true_mask = this_true_mask[
            :, :, kernel_size[0] : -kernel_size[0], kernel_size[1] : -kernel_size[1]
        ]

        if isinstance(this_true_mask, torch.Tensor):
            this_true_mask = this_true_mask.cpu().numpy()
        if isinstance(callback_mask, torch.Tensor):
            callback_mask = callback_mask.cpu().numpy()

        # Check the dimensions
        if callback_mask.shape != this_true_mask.shape:
            msg = (
                f"Callback mask shape {callback_mask.shape} does not match "
                f"true mask shape {this_true_mask.shape}"
            )
            raise ValueError(msg)

        accuracy = np.mean(callback_mask == this_true_mask)
        precision = precision_score(this_true_mask.flatten(), callback_mask.flatten())
        recall = recall_score(this_true_mask.flatten(), callback_mask.flatten())
        f1 = f1_score(this_true_mask.flatten(), callback_mask.flatten())
        dice = 2 * (precision * recall) / (precision + recall + dice_score_epsilon)
        jaccard = jaccard_score(this_true_mask.flatten(), callback_mask.flatten())

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "dice": dice,
            "jaccard": jaccard,
            "percentage": np.mean(callback_mask),
        }

        results.append(
            {
                "name": summary_name,
                **outlier_detection_method,
                **metrics,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "timing": outlier_detection_timing,
            },
        )

        plt.imshow(callback_mask[0, 0], cmap="gray")
        plt.axis("off")
        plt.savefig(
            exp_dir / f"callback_mask_{summary_name.replace(' ', '_')}_"
            f"{outlier_detection_timing}.pdf"
        )
        plt.close()

    # Run the experiment
    if cdl_package == "rosecdl":
        cdl = RoseCDL(
            **cdl_params,
            D_init=d_init,
            outliers_kwargs=outliers_kwargs,
            callbacks=[callback_fn],
        )
        logger.info("Fitting RoseCDL model.")
        cdl.fit(data)
        logger.info("RoseCDL model fitting complete.")
    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

    if i == 0:
        # Plot result dictionary
        fig, axes = plt.subplots(1, cdl.D_hat_.shape[0], figsize=(12, 10))
        for idx, ax in enumerate(axes):
            ax.imshow(cdl.D_hat_[idx, 0], cmap="gray")
            ax.axis("off")
        plot_path = exp_dir / f"D_hat_{summary_name.replace(' ', '_')}.pdf"
        logger.info("Saving estimated dictionary plot to %s", plot_path)
        fig.savefig(plot_path)

    logger.info(
        "Finished run_one for cdl_package=%s, timing=%s, seed=%d. Results length: %d",
        cdl_package,
        outlier_detection_timing,
        seed,
        len(results),
    )

    logger.info(
        "Finished run_one for cdl_package=%s, timing=%s, Last f1 score: %f",
        cdl_package,
        outlier_detection_timing,
        results[-1]["f1"],
    )
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
    parser.add_argument("--reg", type=float, default=0.8, help="Reg parameter")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
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

    exp_dir = Path(args.output) / f"rare_event_2d_reg_{reg}_comparison"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Define base CDL parameters
    cdl_packages = ["rosecdl"]
    cdl_configs = {
        "rosecdl": {
            "kernel_size": (30, 30),
            "n_channels": 1,
            "n_components": 5,
            "lmbd": reg,
            "scale_lmbd": False,
            "epochs": 10 if args.debug else 30,
            "max_batch": 20,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "adam",
            "n_iterations": 40 if args.debug else 60,
            "window": False,
            "device": DEVICE,
            "positive_D": True,
        },
    }

    # Define outlier detection methods
    outlier_detection_methods = (
        [
            {"method": "mad", "alpha": 3.5},
        ]
        if args.debug
        else [
            # {"method": "quantile", "alpha": 0.1},
            # {"method": "zscore", "alpha": 1.5},
            {"method": "mad", "alpha": 3.5},
            {"method": "mad", "alpha": 2.5},
            {"method": "mad", "alpha": 1.5},
            {"method": "mad", "alpha": 4.5},
            {"method": "mad", "alpha": 5.5},


        ]
    )
    for i in range(len(outlier_detection_methods)):
        outlier_detection_methods[i]["opening_window"] = (5, 5)

    outlier_detection_timings = ["during"] if args.debug else ["after", "during"]
    outliers_kwargs = {"moving_average": None, "union_channels": True}

    run_configs = generate_run_config_list(
        cdl_packages=cdl_packages,
        outlier_detection_methods=outlier_detection_methods,
        outlier_detection_timings=outlier_detection_timings,
        cdl_configs=cdl_configs,
        n_runs=n_runs,
        seed=seed,
    )

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
    with (exp_dir / "experiment_config.json").open("w") as f:
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
    results = [r for res in tqdm(results, "Run", total=len(run_configs)) for r in res]

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / f"df_results_{seed}.csv", index=False)

    fig_box, ax_box = plt.subplots(figsize=(8, 6))

    last_epoch = df_results["epoch"].max()
    last_epoch_data = df_results[df_results["epoch"] == last_epoch]

    grouped_data = last_epoch_data.groupby(["name", "timing"])
    boxplot_data = [group["f1"].to_numpy() for _, group in grouped_data]
    tick_labels = [f"{name} ({timing})" for (name, timing), _ in grouped_data]

    box_plot = ax_box.boxplot(boxplot_data, tick_labels=tick_labels, patch_artist=True)

    for i, patch in enumerate(box_plot["boxes"]):
        patch.set_facecolor(f"C{i}")
        patch.set_alpha(0.6)

    ax_box.set_ylabel("F1 Score (Last Epoch)")
    ax_box.set_title("F1 Score for different anomaly detection methods and timings")
    ax_box.tick_params(axis="x", rotation=90)
    ax_box.grid(linestyle="--", alpha=0.7)
    ax_box.set_ylim(0, 1)

    # Save the box plot figure
    plt.tight_layout()
    fig_box.savefig(exp_dir / "anomaly_detection_boxplot.pdf")
