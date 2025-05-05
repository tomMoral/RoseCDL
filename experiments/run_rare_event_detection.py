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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

mem = Memory("__cache__", verbose=0)


def generate_run_config_list(
    cdl_params,
    outlier_detection_methods,
    n_runs,
    master_seed,
):
    """Generate a list of run configurations for the experiment.

    Parameters
    ----------
    cdl_params : dict
        Dictionary containing the parameters for the RoseCDL model.
    outlier_detection_methods : list
        List of outlier detection methods to be used in the experiment.
    n_runs : int, optional
        Number of runs to be performed
    master_seed : int, optional
        Seed for random number generation

    Returns
    -------
    list
        List of dictionaries containing the run configurations.

    """
    rng = np.random.RandomState(master_seed)
    seed = rng.randint(0, 2**32 - 1, size=n_runs)
    run_config_list = []
    for i in range(n_runs):
        for method in outlier_detection_methods:
            cdl_params["outliers_kwargs"] = method
            cdl_params["random_state"] = int(seed[i])
            run_config = {
                "cdl_params": cdl_params,
                "outlier_detection_method": method,
                "seed": int(seed[i]),
                "run_id": i,
                "n_runs": n_runs,
                "master_seed": master_seed,
            }
            run_config_list.append(run_config)
    return run_config_list


@mem.cache
def run_one(
    cdl_params: dict,
    outlier_detection_method: dict,
    seed: int,
    run_id: int,
    n_runs: int,
    master_seed: int,
    data_path: str,
):
    """Run one experiment with the given parameters.

    Parameters
    ----------
    cdl_params : dict
        Dictionary containing the parameters for the RoseCDL model.
    outlier_detection_method : dict
        Dictionary containing the parameters for the outlier detection method.
    seed : int
        Seed for random number generation.
    run_id : int
        ID of the run.
    n_runs : int
        Number of runs to be performed.
    master_seed : int
        Seed for random number generation.
    data_path : str
        Path to the data file.

    Returns
    -------
    dict
        Dictionary containing the results of the experiment.

    """
    logger.info("Running experiment with seed %d", seed)
    logger.info("Run ID: %d", run_id)
    logger.info("Outlier detection method: %s", outlier_detection_method)

    data_params = np.load(data_path)
    data = data_params["X"]
    true_dict = data_params["D"]
    mask = data_params["mask"]

    if data.ndim == 2:
        data = data[None, None, ...]
        mask = mask[None, None, ...]

    # s1 = 990
    # s2 = 550
    # t1 = s1 + 900
    # t2 = s2 + 900
    # data = data[:, :, s1:t1, s2:t2]
    # mask = mask[:, :, s1:t1, s2:t2]

    true_dict = true_dict[:, None, ...]

    init_dict = torch.randn(
        (6, 1, 35, 30), generator=torch.Generator().manual_seed(seed)
    )
    data = torch.tensor(data, device=cdl_params["device"])
    lmbd = cdl_params["lmbd"]

    results = []

    def callback_fn(model, epoch, loss) -> None:
        nonlocal mask
        xh, zh = model.csc(data)

        model_mask_2010 = model.loss_fn.get_outliers_mask(
            xh, zh, data, opening=(20, 10), crop=True
        )
        model_mask_1510 = model.loss_fn.get_outliers_mask(
            xh, zh, data, opening=(15, 10), crop=True
        )

        if isinstance(model_mask_2010, torch.Tensor):
            model_mask_2010 = model_mask_2010.cpu().numpy()
            model_mask_1510 = model_mask_1510.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        model_mask_2010 = model_mask_2010.astype(np.int32)
        mask = mask.astype(np.int32)

        # Cropping the gt mask to match the model mask
        kernel_size = model.csc.kernel_size
        callback_mask = mask[
            :, :, kernel_size[0] : -kernel_size[0], kernel_size[1] : -kernel_size[1]
        ]

        # Compute metrics for model_mask_2010
        precision_2010 = precision_score(
            callback_mask[0, 0].flatten(),
            model_mask_2010[0, 0].flatten(),
            zero_division=0,
        )
        recall_2010 = recall_score(
            callback_mask[0, 0].flatten(),
            model_mask_2010[0, 0].flatten(),
            zero_division=0,
        )
        f1_2010 = f1_score(
            callback_mask[0, 0].flatten(),
            model_mask_2010[0, 0].flatten(),
            zero_division=0,
        )
        iou_2010 = jaccard_score(
            callback_mask.flatten(),
            model_mask_2010.flatten(),
            zero_division=0,
        )

        # Compute metrics for model_mask_1510
        precision_1510 = precision_score(
            callback_mask.flatten(),
            model_mask_1510.flatten(),
            zero_division=0,
        )
        recall_1510 = recall_score(
            callback_mask.flatten(),
            model_mask_1510.flatten(),
            zero_division=0,
        )
        f1_1510 = f1_score(
            callback_mask.flatten(),
            model_mask_1510.flatten(),
            zero_division=0,
        )
        iou_1510 = jaccard_score(
            callback_mask.flatten(),
            model_mask_1510.flatten(),
            zero_division=0,
        )

        logger.info(
            "Epoch %d: loss=%.4f, lmbd=%.4f, precision_2010=%.4f, recall_2010=%.4f, f1_2010=%.4f, iou_2010=%.4f",
            epoch,
            loss,
            lmbd,
            precision_2010,
            recall_2010,
            f1_2010,
            iou_2010,
        )
        logger.info(
            "Epoch %d: precision_1510=%.4f, recall_1510=%.4f, f1_1510=%.4f, iou_1510=%.4f",
            epoch,
            precision_1510,
            recall_1510,
            f1_1510,
            iou_1510,
        )
        logger.info(
            "Number of outliers detected (2010): %d",
            np.sum(model_mask_2010[0, 0]),
        )
        logger.info(
            "Number of outliers detected (1510): %d",
            np.sum(model_mask_1510[0, 0]),
        )
        logger.info(
            "Number of true outliers: %d",
            np.sum(callback_mask[0, 0]),
        )

        # plot the masks
        plt.imshow(model_mask_2010[0, 0])
        plt.imshow(callback_mask[0, 0], alpha=0.5, cmap="gray")
        plt.title(f"Model Mask 2010 - Epoch {epoch}")
        plt.axis("off")
        plt.savefig(
            f"model_mask_2010_epoch_{epoch}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        plt.imshow(model_mask_1510[0, 0])
        plt.imshow(callback_mask[0, 0], alpha=0.5, cmap="gray")
        plt.title(f"Model Mask 1510 - Epoch {epoch}")
        plt.axis("off")
        plt.savefig(
            f"model_mask_1510_epoch_{epoch}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        results.append(
            {
                "run_id": run_id,
                "n_runs": n_runs,
                "master_seed": master_seed,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "lmbd": lmbd,
                "precision_2010": precision_2010,
                "recall_2010": recall_2010,
                "f1_2010": f1_2010,
                "iou_2010": iou_2010,
                "precision_1510": precision_1510,
                "recall_1510": recall_1510,
                "f1_1510": f1_1510,
                "iou_1510": iou_1510,
                **outlier_detection_method,
            }
        )

    cdl = RoseCDL(
        **cdl_params,
        D_init=init_dict,
        callbacks=[callback_fn],
    )
    cdl.fit(data)
    logger.info("Finished training")

    # Plot final dictionary
    fig, ax = plt.subplots(1, len(cdl.D_hat_), figsize=(15, 5))
    for i in range(len(cdl.D_hat_)):
        ax[i].imshow(cdl.D_hat_[i, 0], cmap="gray")
        ax[i].set_title(f"Dictionary Component {i}")
        ax[i].axis("off")
    plt.savefig(
        "learned_dictionary.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    return results


def plot_metric_results(df_method, metric_base, method_name, alpha, epochs, exp_dir):
    """Plot median and quantiles for a given metric (both _2010 and _1510 variants)."""
    fig, ax = plt.subplots()
    colors = {"2010": "blue", "1510": "green"}
    metric_title = metric_base.replace("_", " ").title()
    if metric_base == "iou":
        metric_title = "IoU (Jaccard)"

    for suffix, color in colors.items():
        metric_col = f"{metric_base}_{suffix}"
        if metric_col not in df_method.columns:
            logger.warning(
                "Metric column '%s' not found for method %s. Skipping plot segment.",
                metric_col,
                method_name,
            )
            continue

        quantiles = (
            df_method.groupby("epoch")[metric_col].quantile([0.2, 0.5, 0.8]).unstack()
        )
        q20 = quantiles[0.2]
        median = quantiles[0.5]
        q80 = quantiles[0.8]

        # Plot median and quantiles
        ax.plot(
            epochs,
            median,
            label=f"Median {metric_title} ({suffix.replace('_', ', ')})",
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            epochs,
            q20,
            q80,
            color=color,
            alpha=0.2,
            label=f"20%-80% Quantile Range ({suffix.replace('_', ', ')})",
        )

    ax.set_title(f"{metric_title} - {method_name} (alpha={alpha})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{metric_title} Score")
    ax.legend()
    plot_filename = exp_dir / f"{method_name}_alpha{alpha}_{metric_base}_score.png"
    plt.savefig(plot_filename)
    plt.close(fig)
    logger.info("Saved %s plot to %s", metric_title, plot_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rare event detection experiment.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the data file.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="results",
        help="Directory where the results will be saved.",
    )
    parser.add_argument(
        "--n_runs", "-n", type=int, default=10, help="Number of runs to be performed."
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, help="Number of parallel jobs to be used."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for random number generation."
    )
    parser.add_argument(
        "--reg", type=float, default=0.8, help="Regularization for the RoseCDL model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (n_runs=3, n_jobs=1).",
    )

    args = parser.parse_args()

    if args.debug:
        logging.warning("--- DEBUG MODE ACTIVE ---")
        args.n_runs = 3
        args.n_jobs = 1

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    logger.info("Using device: %s", DEVICE)
    logger.info("Running %d parallel jobs", args.jobs)
    logger.info("Running %d runs", args.n_runs)

    master_seed = (
        np.random.default_rng().integers(0, 2**32 - 1)
        if args.seed is None
        else args.seed
    )
    logger.info("Master seed: %d", master_seed)

    n_runs = args.n_runs

    exp_dir = Path(args.output_dir) / "rare_event_detection"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved in %s", exp_dir)

    cdl_params = {
        "n_components": 6,
        "kernel_size": (30, 25),
        "n_channels": 1,
        "lmbd": args.reg,
        "scale_lmbd": False,
        "epochs": 10 if args.debug else 30,
        "max_batch": 50,
        "mini_batch_size": 10,
        "sample_window": 1000,
        "optimizer": "adam",
        "n_iterations": 5 if args.debug else 65,
        "window": False,
        "device": DEVICE,
        "positive_D": True,
        # Outlier_kwargs will be added in generate_run_config_list
    }

    outlier_detection_methods = (
        [
            {"method": "mad", "alpha": 3.5},
        ]
        if args.debug
        else [
            {"method": "mad", "alpha": 3.5},
            {"method": "zscore", "alpha": 1.5},
            {"method": "iqr", "alpha": 1.5},
            {"method": "quantile", "alpha": 0.1},
        ]
    )
    for i in range(len(outlier_detection_methods)):
        outlier_detection_methods[i]["opening_window"] = (27, 15)

    run_config_list = generate_run_config_list(
        cdl_params=cdl_params,
        outlier_detection_methods=outlier_detection_methods,
        n_runs=n_runs,
        master_seed=master_seed,
    )
    logger.info("Generated %d run configurations", len(run_config_list))
    logger.info("Run configurations: %s", run_config_list)
    logger.info("Starting experiment...")

    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_one)(
            data_path=args.data_path,
            **run_config,
        )
        for run_config in run_config_list
    )

    all_results = [r for res in tqdm(results, total=len(run_config_list)) for r in res]
    logger.info("Finished processing all runs")
    logger.info("Saving results to %s", exp_dir)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(exp_dir / "results.csv", index=False)
    logger.info("Results saved to %s", exp_dir / "results.csv")

    # Plot results
    metrics_to_plot = ["f1", "precision", "recall", "iou"]

    for method_config in outlier_detection_methods:
        method_name = method_config["method"]
        alpha = method_config.get("alpha", "N/A")
        # Filter results for the current method and alpha
        query_parts = [f"`method` == '{method_name}'"]
        if "alpha" in method_config:
            query_parts.append(f"`alpha` == {alpha}")
        query = " & ".join(query_parts)
        method_results = df_results.query(query)

        if method_results.empty:
            logger.warning(
                "No results found for method query: %s. Skipping plots.", query
            )
            continue

        epochs = method_results["epoch"].unique()
        epochs.sort()

        for metric in metrics_to_plot:
            plot_metric_results(
                df_method=method_results,
                metric_base=metric,
                method_name=method_name,
                alpha=alpha,
                epochs=epochs,
                exp_dir=exp_dir,
            )

    logger.info("Finished generating plots.")
