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

        model_mask = model.loss_fn.get_outliers_mask(
            xh, zh, data, opening=outlier_detection_method["opening_window"]
        )

        if isinstance(model_mask, torch.Tensor):
            model_mask = model_mask.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        model_mask = model_mask.astype(np.int32)
        mask = mask.astype(np.int32)

        precision = precision_score(
            mask.flatten(),
            model_mask.flatten(),
            zero_division=0,
        )
        recall = recall_score(
            mask.flatten(),
            model_mask.flatten(),
            zero_division=0,
        )
        f1 = f1_score(
            mask.flatten(),
            model_mask.flatten(),
            zero_division=0,
        )

        iou = jaccard_score(
            mask.flatten(),
            model_mask.flatten(),
            zero_division=0,
        )

        results.append(
            {
                "run_id": run_id,
                "n_runs": n_runs,
                "master_seed": master_seed,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "lmbd": lmbd,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
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

    return results


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
    logger.info("Using device: %d", DEVICE)
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
        "epochs": 5 if args.debug else 30,
        "max_batch": 50,
        "mini_batch_size": 10,
        "sample_window": 1000,
        "optimizer": "adam",
        "n_iterations": 2 if args.debug else 40,
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
            {"method": "quantile", "alpha": 0.05},
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
    for method in outlier_detection_methods:
        method_name = method["method"]
        # Filter results for the current method, and alpha
        query_parts = [f"`method` == '{method_name}'"]
        if "alpha" in method:
            query_parts.append(f"`alpha` == {method['alpha']}")
        query = " & ".join(query_parts)
        method_results = df_results.query(query)

        if method_results.empty:
            logger.warning(
                "No results found for method query: %s. Skipping plots.", query
            )
            continue

        epochs = method_results["epoch"].unique()
        epochs.sort()

        # Calculate quantiles and median for F1 Score
        f1_quantiles = (
            method_results.groupby("epoch")["f1"].quantile([0.2, 0.5, 0.8]).unstack()
        )
        f1_q20 = f1_quantiles[0.2]
        f1_median = f1_quantiles[0.5]
        f1_q80 = f1_quantiles[0.8]

        fig, ax = plt.subplots()
        # Plot median and quantiles
        ax.plot(epochs, f1_median, label="Median F1", color="blue", linewidth=2)
        ax.fill_between(
            epochs,
            f1_q20,
            f1_q80,
            color="blue",
            alpha=0.2,
            label="20%-80% Quantile Range",
        )

        ax.set_title(f"F1 Score - {method_name} (alpha={method.get('alpha', 'N/A')})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.legend()
        plt.savefig(
            exp_dir / f"{method_name}_alpha{method.get('alpha', 'N/A')}_f1_score.png"
        )
        plt.close(fig)
        logger.info("Saved F1 plot for %s", method_name)

        # Calculate quantiles and median for Precision
        precision_quantiles = (
            method_results.groupby("epoch")["precision"]
            .quantile([0.2, 0.5, 0.8])
            .unstack()
        )
        precision_q20 = precision_quantiles[0.2]
        precision_median = precision_quantiles[0.5]
        precision_q80 = precision_quantiles[0.8]

        fig, ax = plt.subplots()
        # Plot median and quantiles
        ax.plot(
            epochs,
            precision_median,
            label="Median Precision",
            color="green",
            linewidth=2,
        )
        ax.fill_between(
            epochs,
            precision_q20,
            precision_q80,
            color="green",
            alpha=0.2,
            label="20%-80% Quantile Range",
        )

        ax.set_title(f"Precision - {method_name} (alpha={method.get('alpha', 'N/A')})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Precision")
        ax.legend()
        plt.savefig(
            exp_dir / f"{method_name}_alpha{method.get('alpha', 'N/A')}_precision.png"
        )
        plt.close(fig)
        logger.info("Saved Precision plot for %s", method_name)

        # Calculate quantiles and median for Recall
        recall_quantiles = (
            method_results.groupby("epoch")["recall"]
            .quantile([0.2, 0.5, 0.8])
            .unstack()
        )
        recall_q20 = recall_quantiles[0.2]
        recall_median = recall_quantiles[0.5]
        recall_q80 = recall_quantiles[0.8]

        fig, ax = plt.subplots()
        # Plot median and quantiles
        ax.plot(epochs, recall_median, label="Median Recall", color="red", linewidth=2)
        ax.fill_between(
            epochs,
            recall_q20,
            recall_q80,
            color="red",
            alpha=0.2,
            label="20%-80% Quantile Range",
        )

        ax.set_title(f"Recall - {method_name} (alpha={method.get('alpha', 'N/A')})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall")
        ax.legend()
        plt.savefig(
            exp_dir / f"{method_name}_alpha{method.get('alpha', 'N/A')}_recall.png"
        )
        plt.close(fig)
        logger.info("Saved Recall plot for %s", method_name)

        # Calculate quantiles and median for IoU
        iou_quantiles = (
            method_results.groupby("epoch")["iou"].quantile([0.2, 0.5, 0.8]).unstack()
        )
        iou_q20 = iou_quantiles[0.2]
        iou_median = iou_quantiles[0.5]
        iou_q80 = iou_quantiles[0.8]

        fig, ax = plt.subplots()
        # Plot median and quantiles
        ax.plot(epochs, iou_median, label="Median IoU", color="purple", linewidth=2)
        ax.fill_between(
            epochs,
            iou_q20,
            iou_q80,
            color="purple",
            alpha=0.2,
            label="20%-80% Quantile Range",
        )

        ax.set_title(
            f"IoU (Jaccard) - {method_name} (alpha={method.get('alpha', 'N/A')})"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU Score")
        ax.legend()
        plt.savefig(
            exp_dir / f"{method_name}_alpha{method.get('alpha', 'N/A')}_iou.png"
        )
        plt.close(fig)
        logger.info("Saved IoU plot for %s", method_name)
