"""Run RoseCDL on multiple datasets.

Evaluate dictionary recovery performance on datasets with varying rare event frequencies,
based on filenames like 'new_text_4_5000_MIND_exo_Z_{freq}.npz'.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
from torch import cuda
from tqdm import tqdm

from rosecdl.loss import OutlierLoss
from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat, get_outliers_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

mem = Memory(location="__cache__", verbose=0)


def get_data_files(
    data_dir: str,
    file_pattern: str = "new_text_4_5000_MIND_exo_Z_*.npz",
    min_freq: float = 0.0,
):
    """Find data files matching the pattern and extract frequencies.

    filtering by min_freq.
    """
    data_path = Path(data_dir)
    files = list(data_path.glob(file_pattern))
    data_configs = []
    # Regex to extract frequency like 0p0700 -> 0.07
    freq_pattern = re.compile(r"_Z_(\d+)p(\d+)\.npz$")
    for f in files:
        match = freq_pattern.search(f.name)
        if match:
            freq_str = f"{match.group(1)}.{match.group(2)}"
            try:
                freq = float(freq_str)
                if freq >= min_freq:
                    data_configs.append({"data_path": str(f), "freq": freq})
                    logging.info("Extracted frequency %s from %s", freq, f.name)
                else:
                    logging.info(
                        "Skipping %s (freq %s < min_freq %s)", f.name, freq, min_freq
                    )
            except ValueError:
                logging.warning("Could not parse frequency from %s", f.name)
        else:
            logging.warning("Could not extract frequency from %s", f.name)
    if not data_configs:
        logging.warning(
            "No files matching pattern '%s' with freq >= %s found in '%s'",
            file_pattern,
            min_freq,
            data_dir,
        )
    return data_configs


def generate_run_config_list(
    data_configs,
    cdl_params,
    sample_window_list,
    reg_list,
    outlier_detection_methods,
    n_runs=1,
    seed=None,
):
    """Generate the list of configurations for the experiment."""
    rng = np.random.default_rng(seed)
    list_seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    run_config_list = []
    for reg_val in reg_list:
        for data_config in data_configs:
            for sample_window in sample_window_list:
                for method in outlier_detection_methods:
                    # Update cdl_params with the current sample_window and reg
                    current_cdl_params = cdl_params.copy()
                    current_cdl_params["sample_window"] = sample_window
                    current_cdl_params["lmbd"] = reg_val  # Set current regularization

                    run_config_list.extend(
                        {
                            "data_path": data_config["data_path"],
                            "freq": data_config["freq"],
                            "cdl_params": current_cdl_params,
                            "sample_window": sample_window,
                            "outlier_detection_method": method,
                            "reg": reg_val,
                            "seed": int(s),
                            "i": i,
                        }
                        for i, s in enumerate(list_seeds)
                    )
    return run_config_list


@mem.cache
def run_one(
    data_path: str,
    freq: float,
    cdl_params: dict,
    outlier_detection_method: dict,
    seed: int,
    i: int,
    reg: float,
    exp_dir: Path,
    **kwargs,
):
    """Run the experiment for a given data file (frequency)."""
    logging.info(
        "Running freq=%.4f, reg=%s, sw=%s with %s (run %d, seed=%s) from %s",
        freq,
        reg,
        cdl_params["sample_window"],
        outlier_detection_method,
        i + 1,
        seed,
        data_path,
    )

    if outlier_detection_method["method"] == "none":
        outliers_kwargs = None
    else:
        outliers_kwargs = outlier_detection_method.copy()

    D_exo_reshaped = None
    try:
        data_params = np.load(data_path)
        X = data_params["X"]
        D_true = data_params["D"]
        D_exo = data_params.get("D_exo")
        true_mask = data_params["mask"]

        # Reshape D_exo if it exists
        if D_exo is not None:
            D_exo_reshaped = np.expand_dims(D_exo, axis=1)
            if D_exo_reshaped.ndim != 4:
                raise ValueError(
                    f"D_exo_reshaped should be 4D, but got {D_exo_reshaped.ndim}D"
                )

    except FileNotFoundError:
        logging.exception("Data file not found at %s", data_path)
        return []
    except KeyError as e:
        logging.exception("Missing key %s in %s", e, data_path)
        return []

    X = X[None, None, ...]
    if X.ndim != 4:
        raise ValueError(f"X should be 4D, but got {X.ndim}D")

    true_mask = true_mask[None, None, ...]
    if true_mask.ndim != 4:
        raise ValueError(f"Mask should be 4D, but got {true_mask.ndim}D")

    D_true = np.expand_dims(D_true, axis=1)  # Add channel dimension
    if D_true.ndim != 4:
        raise ValueError(f"D_true should be 4D, but got {D_true.ndim}D")

    summary_name = f"freq_{freq:.4f}_reg_{reg}_run_{i}"
    summary_name += "{method} (alpha={alpha:.02f})".format(
        **outlier_detection_method,
    )

    # ========== Callback ==========
    results = []
    t_start_global = time.perf_counter()

    def callback_fn(model, epoch, loss):
        nonlocal t_start_global
        runtime = time.perf_counter() - t_start_global
        recovery_score = evaluate_D_hat(D_true, model.D_hat_)
        # Calculate exo score if D_exo exists
        exo_score = None
        if D_exo_reshaped is not None:
            exo_score = evaluate_D_hat(D_exo_reshaped, model.D_hat_)

        if isinstance(model.loss_fn, OutlierLoss):  # Using OD
            metrics = get_outliers_metric(
                true_outliers_mask=true_mask,
                rosecdl=model,
                X=X,
                crop=True,
            )
        else:
            metrics = {}

        results.append(
            {
                "name": summary_name,
                "freq": freq,
                "reg": reg,  # Add reg here
                "recovery_score": recovery_score,
                "exo_score": exo_score,  # Add exo_score here
                "seed": seed,
                "run_index": i,
                "epoch": epoch,
                "loss": loss,
                "time": runtime,
                "sample_window": cdl_params["sample_window"],
                **outlier_detection_method,
                **metrics,
            }
        )
        t_start_global = time.perf_counter()

    # Initializing D_init randomly for each run with the same seed
    rng_init = torch.Generator().manual_seed(seed)
    D_init = torch.randn(
        (
            cdl_params["n_components"],
            cdl_params["n_channels"],
            *cdl_params["kernel_size"],
        ),
        generator=rng_init,
    )
    if cdl_params.get("positive_D", False):
        D_init = torch.abs(D_init)

    cdl = RoseCDL(
        **cdl_params,
        D_init=D_init,
        outliers_kwargs=outliers_kwargs,
        callbacks=[callback_fn],
    )

    try:
        cdl.fit(X)
    except Exception as e:
        logging.exception(
            "Error during RoseCDL fit for freq=%s, reg=%s, run=%s",
            freq,
            reg,
            i,
        )
        results.append(
            {
                "name": summary_name,
                "freq": freq,
                "reg": reg,  # Add reg here
                "seed": seed,
                "run_index": i,
                "error": str(e),
                "sample_window": cdl_params["sample_window"],
            }
        )
        return results

    if i == 0:  # Plot dictionary only for the first run of each frequency
        try:
            # Plot true dictionary
            fig_true, axes_true = plt.subplots(1, D_true.shape[0], figsize=(12, 3))
            if D_true.shape[0] == 1:
                axes_true = [axes_true]  # Handle single atom case
            fig_true.suptitle(f"True Dictionary (Freq: {freq:.4f})")
            for idx, ax in enumerate(axes_true):
                ax.imshow(D_true[idx, 0], cmap="gray")
                ax.axis("off")
            fig_true.savefig(
                exp_dir / f"dict_true_freq_{freq:.4f}.pdf"
            )  # Keep filename general for true dict
            plt.close(fig_true)

            # Plot result dictionary
            fig_res, axes_res = plt.subplots(1, cdl.D_hat_.shape[0], figsize=(12, 3))
            if cdl.D_hat_.shape[0] == 1:
                axes_res = [axes_res]  # Handle single atom case
            fig_res.suptitle(
                f"Recovered Dictionary (Freq: {freq:.4f}, Reg: {reg}, Run: {i})"
            )  # Add reg to title
            for idx, ax in enumerate(axes_res):
                ax.imshow(cdl.D_hat_[idx, 0], cmap="gray")
                ax.axis("off")
            fig_res.savefig(
                exp_dir / f"D_hat_freq_{freq:.4f}_reg_{reg}_run_{i}.pdf"
            )  # Add reg to filename
            plt.close(fig_res)
        except Exception as plot_err:
            logging.warning(
                "Plotting failed for freq=%s, reg=%s, run=%s: %s",
                freq,
                reg,
                i,
                plot_err,
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run RoseCDL dictionary recovery experiments for multiple"
            " frequencies and regularizations."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/",
        help="Directory containing the NPZ data files.",
    )
    parser.add_argument(
        "--n-jobs", "-j", type=int, default=1, help="Number of parallel jobs."
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Master seed for reproducible job generation.",
    )
    parser.add_argument(
        "--n-runs",
        "-n",
        type=int,
        default=5,
        help="Number of repetitions for each frequency.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory to store the results.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (n_runs=1).",
    )
    args = parser.parse_args()

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    logging.info("Using device: %s", DEVICE)

    rng = np.random.default_rng(args.seed)
    master_seed = rng.integers(0, 2**32 - 1) if args.seed is None else args.seed
    logging.info("Master Seed: %s", master_seed)

    n_runs = args.n_runs

    reg_list = [0.2, 0.5, 0.9]
    logging.info("Testing regularization values: %s", reg_list)

    if args.debug:
        logging.warning("--- DEBUG MODE ACTIVE: Setting n_runs to 1 ---")
        n_runs = 1
        args.n_jobs = 1
        reg_list = [reg_list[0]]
        logging.warning("--- DEBUG MODE ACTIVE: Using only reg=%s ---", reg_list[0])

    exp_dir = Path(args.output) / "OD_multi_freq_reg_window_experiment"
    exp_dir.mkdir(exist_ok=True, parents=True)
    logging.info("Saving results to: %s", exp_dir)

    cdl_params = {
        "n_components": 8,
        "kernel_size": (30, 25),
        "n_channels": 1,
        "scale_lmbd": False,
        "epochs": 5 if args.debug else 30,
        "max_batch": 40,
        "mini_batch_size": 10,
        "sample_window": -1,  # Will be set in `generate_run_config_list`
        "optimizer": "adam",
        "n_iterations": 2 if args.debug else 40,
        "window": False,
        "device": DEVICE,
        "positive_D": True,
        # random_state will be set per run
    }

    try:
        # Pass min_freq if needed, e.g., from args or a fixed value
        # For now, using the default min_freq=0.0
        data_configs = get_data_files(args.data_dir, min_freq=0.005)
    except FileNotFoundError:
        logging.exception("Data directory not found.")
        sys.exit(1)

    logging.info("Found %s data configurations:", len(data_configs))
    for cfg in data_configs:
        logging.info("  - Freq: %.4f, Path: %s", cfg["freq"], cfg["data_path"])

    sample_window_list = [1000]  # , 1500]
    sample_window_list = [(sw, sw) for sw in sample_window_list]

    outlier_detection_methods = (
        [
            {"method": "mad", "alpha": 3.5},
        ]
        if args.debug
        else [
            {"method": "none", "alpha": -1},
            {"method": "quantile", "alpha": 0.1},
            {"method": "quantile", "alpha": 0.05},
            {"method": "mad", "alpha": 3.5},
        ]
    )
    for i in range(len(outlier_detection_methods)):
        outlier_detection_methods[i]["opening_window"] = (27, 15)

    run_configs = generate_run_config_list(
        data_configs=data_configs,
        cdl_params=cdl_params,
        outlier_detection_methods=outlier_detection_methods,
        n_runs=n_runs,
        sample_window_list=sample_window_list,
        reg_list=reg_list,
        seed=master_seed,
    )

    logging.info("Generated %s total run configurations.", len(run_configs))

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            exp_dir=exp_dir,
            **rc,
        )
        for rc in run_configs
    )

    # Flatten results list
    all_results = [
        item
        for sublist in tqdm(results, "Collecting results", total=len(run_configs))
        if sublist
        for item in sublist
    ]

    if not all_results:
        logging.warning("No results were generated. Exiting.")
        sys.exit()

    # Save results
    df_results = pd.DataFrame(all_results)
    # Handle potential errors recorded
    if "error" in df_results.columns:
        logging.warning("\nErrors occurred during some runs:")
        logging.warning(
            df_results[df_results["error"].notna()][
                ["freq", "reg", "sample_window", "run_index", "seed", "error"]
            ]
        )
        # Filter out errored runs for plotting if necessary
        df_plot = df_results[df_results["error"].isna()].copy()
    else:
        df_plot = df_results.copy()

    results_filename = (
        exp_dir / "df_results_multi_freq_reg_window.csv"
    )  # General filename
    df_results.to_csv(results_filename, index=False)
    logging.info("Results saved to %s", results_filename)

    # Plotting section - loop over unique reg values
    if not df_plot.empty:
        unique_regs = sorted(df_plot["reg"].unique())
        logging.info("Generating plots for reg values: %s", unique_regs)

        for reg_val in unique_regs:
            df_reg = df_plot[df_plot["reg"] == reg_val].copy()
            logging.info("Plotting for reg = %s (%s data points)", reg_val, len(df_reg))

            if df_reg.empty:
                logging.warning(
                    "Skipping plots for reg=%s as no data is available.", reg_val
                )
                continue

            # Plot final recovery score vs frequency for this reg
            if "recovery_score" in df_reg.columns:
                final_scores = df_reg.loc[
                    df_reg.groupby(["freq", "run_index", "sample_window"])[
                        "epoch"
                    ].idxmax()
                ]  # Group by sample_window too

                plt.figure(figsize=(12, 7))  # Wider figure for sample_window hue
                import seaborn as sns

                sns.boxplot(
                    data=final_scores,
                    x="freq",
                    y="recovery_score",
                    hue="sample_window",
                    palette="viridis",
                )

                plt.xlabel("Frequency of Rare Event")
                plt.ylabel("Final Recovery Score (vs True D)")
                plt.title(
                    f"Dictionary Recovery Score vs Frequency (Reg={reg_val}, Runs={n_runs})"
                )
                plt.legend(
                    title="Sample Window", bbox_to_anchor=(1.05, 1), loc="upper left"
                )
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plot_filename = exp_dir / f"final_recovery_vs_freq_reg_{reg_val}.pdf"
                plt.savefig(plot_filename)
                logging.info("Final score plot saved to %s", plot_filename)
                plt.close()

            # Plot final exo score vs frequency for this reg
            if "exo_score" in df_reg.columns and df_reg["exo_score"].notna().any():
                final_scores_exo = df_reg.loc[
                    df_reg.groupby(["freq", "run_index", "sample_window"])[
                        "epoch"
                    ].idxmax()
                ]  # Recalculate or reuse final_scores if grouping is same

                plt.figure(figsize=(12, 7))  # Wider figure for sample_window hue
                sns.boxplot(
                    data=final_scores_exo,
                    x="freq",
                    y="exo_score",
                    hue="sample_window",
                    palette="viridis",
                )

                plt.xlabel("Frequency of Rare Event")
                plt.ylabel("Final Recovery Score (vs Exo D)")
                plt.title(
                    f"Exogenous Dictionary Recovery Score vs Frequency "
                    f"(Reg={reg_val}, Runs={n_runs})"
                )
                plt.legend(
                    title="Sample Window", bbox_to_anchor=(1.05, 1), loc="upper left"
                )
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                exo_plot_filename = (
                    exp_dir / f"final_exo_score_vs_freq_reg_{reg_val}.pdf"
                )
                plt.savefig(exo_plot_filename)
                logging.info("Final exo score plot saved to %s", exo_plot_filename)
                plt.close()
            elif "exo_score" in df_reg.columns:
                logging.warning(
                    "Skipping exo_score plot for reg=%s as all values are NaN.", reg_val
                )

            # Recovery score curves over epochs, runs per frequency for this reg
            if "recovery_score" in df_reg.columns:
                plt.figure(figsize=(14, 8))  # Wider figure for sample_window facet/hue

                # Use seaborn FacetGrid for clarity across sample_windows
                # Calculate median and quantiles per freq, epoch, sample_window
                curves = (
                    df_reg.groupby(["freq", "epoch", "sample_window"])["recovery_score"]
                    .quantile([0.2, 0.5, 0.8])
                    .unstack()
                    .reset_index()
                )
                curves = curves.rename(columns={0.5: "median", 0.2: "q20", 0.8: "q80"})

                g = sns.FacetGrid(
                    curves,
                    col="sample_window",
                    col_wrap=3,
                    hue="freq",
                    palette="viridis",
                    sharey=True,
                    height=4,
                )
                g.map_dataframe(sns.lineplot, x="epoch", y="median")
                # Add shaded areas (might require iterating through axes)
                for ax in g.axes_dict.values():
                    sw = ax.get_title().split(" = ")[
                        -1
                    ]  # Extract sample_window from title
                    try:
                        sw_val = tuple(
                            map(int, sw.strip("()").split(", "))
                        )  # Convert string tuple back
                    except:  # Handle potential parsing errors if title format changes
                        sw_val = sw  # Keep as string if parsing fails

                    curves_sw = curves[curves["sample_window"] == sw_val]
                    unique_freqs_sw = sorted(curves_sw["freq"].unique())
                    colors_sw = plt.cm.viridis(np.linspace(0, 1, len(unique_freqs_sw)))
                    for i, freq in enumerate(unique_freqs_sw):
                        freq_curve = curves_sw[curves_sw["freq"] == freq]
                        ax.fill_between(
                            freq_curve["epoch"],
                            freq_curve["q20"],
                            freq_curve["q80"],
                            alpha=0.2,
                            color=colors_sw[i],
                        )

                g.set_axis_labels("Epoch", "Median Recovery Score")
                g.set_titles(col_template="Sample Window = {col_name}")
                g.add_legend(title="Frequency")
                g.fig.suptitle(
                    f"Recovery Score during Training (Reg={reg_val}, Runs={n_runs})",
                    y=1.02,
                )  # Adjust title position
                g.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout

                curve_plot_filename = (
                    exp_dir / f"recovery_curves_freq_reg_{reg_val}.pdf"
                )
                plt.savefig(curve_plot_filename)
                logging.info("Recovery curve plot saved to %s", curve_plot_filename)
                plt.close()

    else:
        logging.warning("Skipping plotting as no valid results data is available.")

    logging.info("--- Experiment Finished ---")