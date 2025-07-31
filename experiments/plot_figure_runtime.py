import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

# =============================================================================
# Set global parameters for NeurIPS style plots
# =============================================================================
plt.rcParams.update(
    {
        "font.size": 9,  # Base font size
        "axes.labelsize": 7,  # Font size for x and y labels
        "axes.titlesize": 7,  # Font size for the title
        "xtick.labelsize": 7,  # Font size for x-axis tick labels
        "ytick.labelsize": 7,  # Font size for y-axis tick labels
        "legend.fontsize": 8,  # Font size for the legend
        "lines.linewidth": 1.5,  # Linewidth for plot lines
        "pdf.fonttype": 42,  # Embed fonts in PDF for submission
        "text.usetex": True,  # Enable LaTeX rendering for text
        "axes.labelpad": 1.8,  # Space between axis ticks and axis labels
        "xtick.major.pad": 1.5,  # Reduce space between x-tick and label
        "ytick.major.pad": 1.5,  # Reduce space between y-tick and label
    }
)

ALPHA_FILL = 0.1


def plot_combined_figure(
    time_grid,
    all_interp_train_losses_1d,
    all_interp_train_losses_2d,
    df_recovery,
    output_dir,
    boxplot_recovery=False,  # New argument to control boxplot vs. curve
):
    """Plot combined figure with 1D runtime, 2D runtime, and recovery plots."""
    # Name mapping dictionary
    name_mapping = {
        "alphacsc": r"AlphaCSC",
        "deepcdl": r"DeepCDL",
        "rosecdl": r"\textbf{RoseCDL}",
        "sporco": r"Sporco",
    }

    # Create figure with GridSpec
    fig = plt.figure(figsize=(5.5, 2))
    gs = plt.GridSpec(
        2,
        3,
        height_ratios=[0.1, 1],
        left=0.07,
        right=0.98,
        top=1,
        bottom=0.2,
        hspace=0.6,
        wspace=0.3,
    )

    # Create axes for the three plots
    ax_legend = fig.add_subplot(gs[0, :])
    ax_1d = fig.add_subplot(gs[1, 0])
    ax_2d = fig.add_subplot(gs[1, 1])
    ax_recovery = fig.add_subplot(gs[1, 2])

    # Turn off legend axis
    ax_legend.set_axis_off()

    # Create a color dictionary for consistent colors
    all_algorithms = set()
    for name in (
        list(all_interp_train_losses_1d.keys())
        + list(all_interp_train_losses_2d.keys())
        + list(df_recovery["name"].unique())
    ):
        all_algorithms.add(name)

    # Create color map
    colors = plt.cm.tab10(range(len(all_algorithms)))
    color_dict = dict(zip(sorted(all_algorithms), colors, strict=False))

    all_lines = []
    all_labels = []

    # Plot 1D runtime data
    for name in all_interp_train_losses_1d:
        curves = all_interp_train_losses_1d[name]
        median_curve = np.median(curves, axis=0)
        q02_curve = np.quantile(curves, 0.2, axis=0)
        q8_curve = np.quantile(curves, 0.8, axis=0)

        color = color_dict[name]
        display_name = name_mapping.get(name, name)  # Get display name or use original
        (line,) = ax_1d.plot(time_grid, median_curve, label=display_name, color=color)
        ax_1d.fill_between(
            time_grid, q02_curve, q8_curve, alpha=ALPHA_FILL, color=color
        )

        if display_name not in all_labels:
            all_lines.append(line)
            all_labels.append(display_name)

    # Plot 2D runtime data
    for name in all_interp_train_losses_2d:
        curves = all_interp_train_losses_2d[name]
        median_curve = np.median(curves, axis=0)
        q02_curve = np.quantile(curves, 0.2, axis=0)
        q8_curve = np.quantile(curves, 0.8, axis=0)

        color = color_dict[name]
        display_name = name_mapping.get(name, name)  # Get display name or use original
        (line,) = ax_2d.plot(time_grid, median_curve, label=display_name, color=color)
        ax_2d.fill_between(
            time_grid, q02_curve, q8_curve, alpha=ALPHA_FILL, color=color
        )

        if display_name not in all_labels:
            all_lines.append(line)
            all_labels.append(display_name)

    # Plot recovery data
    if not boxplot_recovery:
        quantiles = df_recovery.groupby(["name", "epoch"])["recovery_score"].quantile(
            [0.2, 0.5, 0.8]
        )
        quantiles = quantiles.reset_index().pivot_table(
            index=["name", "epoch"], columns="level_2", values="recovery_score"
        )

        for name in df_recovery["name"].unique():
            if name not in quantiles.index.get_level_values("name"):
                continue
            epochs = quantiles.loc[name].index
            q20 = quantiles.loc[name][0.2]
            q50 = quantiles.loc[name][0.5]
            q80 = quantiles.loc[name][0.8]

            color = color_dict[name]
            display_name = name_mapping.get(name, name)  # Get name or use original
            (line,) = ax_recovery.plot(epochs, q50, label=display_name, color=color)
            ax_recovery.fill_between(epochs, q20, q80, alpha=ALPHA_FILL, color=color)

            if display_name not in all_labels:
                all_lines.append(line)
                all_labels.append(display_name)
        ax_recovery.set_xlabel("Epoch")
        ax_recovery.set_ylabel("Recovery Score (2D)")
    else:
        # Boxplot of recovery at last epoch for each solver
        last_epochs = df_recovery.groupby("name")["epoch"].max()
        data = []
        xticklabels = []
        box_colors = []
        for idx, name in enumerate(df_recovery["name"].unique()):
            last_epoch = last_epochs[name]
            scores = df_recovery.loc[
                (df_recovery["name"] == name) & (df_recovery["epoch"] == last_epoch),
                "recovery_score",
            ]
            if scores.empty:
                continue
            data.append(scores)
            xticklabels.append(name_mapping.get(name, name))
            box_colors.append(color_dict[name])
        # Draw boxplot
        box = ax_recovery.boxplot(
            data,
            patch_artist=True,
            tick_labels=xticklabels,
            medianprops={"color": "black"},
        )
        # Set box colors
        for patch, color in zip(box["boxes"], box_colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax_recovery.set_xlabel("Solver")
        ax_recovery.set_ylabel("Recovery Score")

    # Configure axes

    ax_1d.set_xscale("log")
    ax_1d.set_yscale("log")
    ax_1d.set_xlabel("Time (s)")
    ax_1d.set_ylabel("Test Loss (1D)")

    # Only show labels on major ticks (e.g., 10^5, not 2*10^5)
    ax_1d.xaxis.set_major_locator(LogLocator(base=10.0))
    ax_1d.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_1d.xaxis.set_minor_formatter(NullFormatter())
    ax_1d.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_1d.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_1d.yaxis.set_minor_formatter(NullFormatter())

    ax_2d.set_xscale("log")
    ax_2d.set_yscale("log")
    ax_2d.set_xlabel("Time (s)")
    ax_2d.set_ylabel("Test Loss (2D)")

    # Create shared legend in top row
    ax_legend.legend(
        all_lines,
        all_labels,
        loc="upper center",
        ncol=len(all_labels),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save the figure
    if boxplot_recovery:
        plt.savefig(output_dir / "combined_plots_boxplot.pdf", format="pdf")
    else:
        plt.savefig(output_dir / "combined_plots.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    # =============================================================================
    # Argument Parsing
    # =============================================================================
    parser = argparse.ArgumentParser(
        description="Plot results from runtime comparison experiment"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="plots",
        help="Directory to save the plots",
    )
    # ==================================================================
    # 1D Time
    # ==================================================================
    parser.add_argument("--time-1d", type=str, required=True, help="Dir of 1D time csv")
    parser.add_argument(
        "--separate-1d",
        action="store_true",
        help="If set, load results from separate df_results_<name>.csv "
        "files instead of a single df_results.csv",
    )
    parser.add_argument(
        "--solver-1d",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Specify one or more solver names to plot for 1D data. "
            "If not provided, all solvers found will be plotted."
        ),
    )
    # =================================================================
    # 2D Time
    # =================================================================
    parser.add_argument("--time-2d", type=str, required=True, help="Dir of 2D time csv")
    parser.add_argument(
        "--separate-2d",
        action="store_true",
        help="If set, load results from separate df_results_<name>.csv "
        "files instead of a single df_results.csv",
    )
    parser.add_argument(
        "--solver-2d",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Specify one or more solver names to plot for 2D data. "
            "If not provided, all solvers found will be plotted."
        ),
    )
    # =================================================================
    # RECOVERY
    # =================================================================
    parser.add_argument(
        "--recovery",
        type=str,
        default=None,
        help=("Directory containing recovery results. "),
    )
    parser.add_argument(
        "--separate-recovery",
        action="store_true",
        help="If set, load results from separate df_results_<name>.csv "
        "files instead of a single df_results.csv",
    )
    parser.add_argument(
        "--solver-recover",
        type=str,
        nargs="+",
        default=None,
        help="Filter by specific solver names for recovery plots",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Set output directories for 1D and 2D plots to avoid overwriting
    output_dir_1d = output_dir / "plots_1d"
    output_dir_2d = output_dir / "plots_2d"
    output_dir_1d.mkdir(parents=True, exist_ok=True)
    output_dir_2d.mkdir(parents=True, exist_ok=True)

    # ==========================      1D      =====================================
    # =============================================================================
    # Data Loading
    # =============================================================================

    # Use the directory from --time-1d for loading CSVs
    results_dir_1d = Path(args.time_1d)

    if args.separate_1d:
        all_dfs = []
        file_pattern = results_dir_1d / "df_results_*.csv"
        csv_files = list(results_dir_1d.glob("df_results_*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

        print(f"Found {len(csv_files)} separate result files:")
        for filepath in csv_files:
            print(f"  - Loading {filepath.name}")
            try:
                # Extract method name from filename: df_results_['methodname'].csv
                method_name = (
                    filepath.stem.split("_")[-1] if "_" in filepath.stem else None
                )
                # This gives us "methodname" from "['methodname']"
                method_name = method_name.strip("[]'") if method_name else None

                if not method_name:
                    print(
                        "Warning: Could not extract method name from "
                        f"{filepath.name}. Skipping."
                    )
                    continue

                # If specific solvers are requested, only load those files
                if args.solver_1d is not None and method_name not in args.solver_1d:
                    print(f"    Skipping {filepath.name} (not in requested solvers).")
                    continue

                df_method = pd.read_csv(filepath)
                df_method["name"] = method_name
                all_dfs.append(df_method)
            except pd.errors.EmptyDataError as e:
                print(f"    Error loading or processing {filepath.name}: {e}")
            except FileNotFoundError as e:
                print(f"    Error loading or processing {filepath.name}: {e}")

        if not all_dfs:
            raise ValueError(
                "No valid data loaded from separate CSV files (check filters?)."
            )

        df_results = pd.concat(all_dfs, ignore_index=True)
        print("Successfully combined data from separate files.")
    else:
        single_csv_path = results_dir_1d / "df_results.csv"
        if not single_csv_path.is_file():
            raise FileNotFoundError(f"Single results file not found: {single_csv_path}")
        print(f"Loading results from single file: {single_csv_path}")
        df_results = pd.read_csv(single_csv_path)

        # Filter based on --solver-1d argument if provided
        if args.solver_1d is not None:
            print(f"Filtering results for solvers: {', '.join(args.solver_1d)}")
            original_rows = len(df_results)
            df_results = df_results[df_results["name"].isin(args.solver_1d)]
            print(f"  Kept {len(df_results)} out of {original_rows} rows.")
            if df_results.empty:
                print(
                    "Warning: No data remaining after filtering for specified solvers."
                )

    # =============================================================================
    # Data Preprocessing
    # =============================================================================
    if not df_results.empty:
        # Adjust time for alphacsc epoch 0 if alphacsc is in the data
        if "alphacsc" in df_results["name"].unique():
            df_results.loc[
                (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0), "time"
            ] = 1e-1

            # Add the first loss evaluation of alphacsc to other methods for same start
            epoch0alphacsc = df_results[
                (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0)
            ].copy()

            all_dfs_preproc = [df_results]
            original_names = df_results["name"].unique()
            for name in original_names:
                if name == "alphacsc":
                    continue
                epoch0_copy = epoch0alphacsc.copy()
                epoch0_copy["name"] = name
                df_results.loc[df_results["name"] == name, "epoch"] += 1
                all_dfs_preproc.append(epoch0_copy)

            df_results = pd.concat(all_dfs_preproc, ignore_index=True)
        else:
            print("alphacsc not found in the data, skipping epoch 0 adjustment.")

    # =============================================================================
    #   Plotting RUNTIME 1D
    # =============================================================================

    if df_results.empty:
        print("Warning: No data to plot after filtering.")
        sys.exit()

    # Ensure data is sorted for cumulative sum and interpolation
    df_results = df_results.sort_values(by=["name", "seed", "epoch"])

    # Precompute cumulative time and adjusted losses
    # Check if 'time' column exists and has data before cumsum
    if "time" not in df_results.columns or df_results["time"].isna().all():
        print(
            "Warning: 'time' column missing or empty. Cannot compute cumulative time."
        )
        sys.exit()

    df_results["cum_time"] = df_results.groupby(["name", "seed"])["time"].cumsum()

    # Check if loss columns exist before proceeding
    if (
        "loss_true" not in df_results.columns
        or "test_loss_true" not in df_results.columns
    ):
        print(
            "Warning: 'loss_true' or 'test_loss_true' columns missing. "
            "Cannot plot losses."
        )
        sys.exit()

    min_loss_true = df_results["loss_true"].min()
    min_test_loss_true = df_results["test_loss_true"].min()
    # Add a small epsilon to avoid issues with log scale if min loss is 0
    epsilon = 1e-1
    df_results["loss_true_adj"] = df_results["loss_true"] - min_loss_true + epsilon
    df_results["test_loss_true_adj"] = (
        df_results["test_loss_true"] - min_test_loss_true + epsilon
    )

    # Determine global time range and create a common logarithmic time grid
    # Ensure min_time is strictly positive for logspace
    valid_times = df_results[df_results["cum_time"] > 0]["cum_time"]
    if valid_times.empty:
        print(
            "Warning: No positive cumulative time values found. "
            "Cannot create time grid."
        )
        sys.exit()
    min_time = max(valid_times.min(), 1e-4)
    max_time = df_results["cum_time"].max()
    if min_time >= max_time:
        print(
            "Warning: Invalid time range for interpolation "
            f"({min_time=}, {max_time=}). Skipping plots."
        )
        sys.exit()
    n_grid_points = 500  # Number of points for interpolation grid
    time_grid = np.logspace(np.log10(min_time), np.log10(max_time), n_grid_points)

    # =============================================================================
    # Plotting Train Loss
    # =============================================================================
    plt.figure(figsize=(3.5, 2.2))
    ax_train = plt.gca()
    all_interp_train_losses = {}
    plotted_names_train = []  # Keep track of names plotted

    for name in df_results.name.unique():
        interp_train_losses_name = []
        df_name = df_results[df_results.name == name]
        for seed in df_name.seed.unique():
            df_run = df_name[df_name.seed == seed].copy()
            # Ensure time is monotonically increasing for interpolation
            df_run = df_run.drop_duplicates(subset=["cum_time"], keep="first")
            run_cum_time = df_run["cum_time"].to_numpy()
            run_loss_true = df_run["loss_true_adj"].to_numpy()

            # Skip runs with insufficient points for interpolation
            if len(run_cum_time) < 2:
                continue

            # Ensure data is valid for log transformation and interpolation
            valid_mask = (run_cum_time > 0) & (run_loss_true > 0)
            if np.sum(valid_mask) < 2:  # Need at least two points for interpolation
                continue

            current_times = run_cum_time[valid_mask]
            current_losses = run_loss_true[valid_mask]

            # Prepare data for log-log interpolation
            log_current_times = np.log(current_times)
            log_current_losses = np.log(current_losses)

            # Sort by log_current_times as np.interp requires xp to be increasing
            sort_indices = np.argsort(log_current_times)
            sorted_log_times = log_current_times[sort_indices]
            sorted_log_losses = log_current_losses[sort_indices]

            unique_log_times, unique_indices = np.unique(
                sorted_log_times, return_index=True
            )
            if len(unique_log_times) < 2:
                continue
            unique_log_losses = sorted_log_losses[unique_indices]

            # Perform interpolation in log-log scale
            # np.log(time_grid) are the target x-points (log of actual time values
            # on grid)
            # unique_log_times are the data x-points (log scale, sorted, unique)
            # unique_log_losses are the data y-points (log scale,
            # corresponding to unique_log_times)
            log_interp_values = np.interp(
                np.log(time_grid), unique_log_times, unique_log_losses
            )
            interp_train = np.exp(log_interp_values)
            interp_train_losses_name.append(interp_train)

        # Store all interpolated curves for the method
        if interp_train_losses_name:
            all_interp_train_losses[name] = np.array(interp_train_losses_name)

        # Calculate median and quantiles across seeds
        if (
            name in all_interp_train_losses
            and all_interp_train_losses[name].shape[0] > 0
        ):
            median_curve = np.median(all_interp_train_losses[name], axis=0)
            q02_curve = np.quantile(all_interp_train_losses[name], 0.2, axis=0)
            q8_curve = np.quantile(all_interp_train_losses[name], 0.8, axis=0)

            # Plot median curve and shaded quantile region
            (line,) = ax_train.plot(time_grid, median_curve, label=name)
            color = line.get_color()
            ax_train.fill_between(
                time_grid, q02_curve, q8_curve, color=color, alpha=ALPHA_FILL
            )
            plotted_names_train.append(name)

    if not plotted_names_train:
        print("Warning: No data plotted for Train Loss.")
    else:
        ax_train.set_xscale("log")
        ax_train.set_yscale("log")
        ax_train.set_xlabel("Time (s)")
        ax_train.set_ylabel("Train Loss")
        ax_train.legend()
        plt.tight_layout()
        plt.savefig(output_dir_1d / "corrected_interp1d_loss_true.pdf", format="pdf")
    plt.close()

    # =============================================================================
    # Plotting Test Loss
    # =============================================================================
    plt.figure(figsize=(3.5, 2.2))
    ax_test = plt.gca()
    all_interp_test_losses = {}
    plotted_names_test = []

    for name in df_results.name.unique():
        interp_test_losses_name = []
        df_name = df_results[df_results.name == name]
        for seed in df_name.seed.unique():
            df_run = df_name[df_name.seed == seed].copy()
            # Ensure time is monotonically increasing for interpolation
            df_run = df_run.drop_duplicates(subset=["cum_time"], keep="first")
            run_cum_time = df_run["cum_time"].to_numpy()
            run_test_loss_true = df_run["test_loss_true_adj"].to_numpy()

            # Skip runs with insufficient points for interpolation
            if len(run_cum_time) < 2:
                continue

            # Ensure data is valid for log transformation and interpolation
            valid_mask = (run_cum_time > 0) & (run_test_loss_true > 0)
            if np.sum(valid_mask) < 2:  # Need at least two points for interpolation
                continue

            current_times = run_cum_time[valid_mask]
            current_losses = run_test_loss_true[valid_mask]

            # Prepare data for log-log interpolation
            log_current_times = np.log(current_times)
            log_current_losses = np.log(current_losses)

            # Sort by log_current_times as np.interp requires xp to be increasing
            sort_indices = np.argsort(log_current_times)
            sorted_log_times = log_current_times[sort_indices]
            sorted_log_losses = log_current_losses[sort_indices]

            # Ensure uniqueness of sorted_log_times for np.interp
            unique_log_times, unique_indices = np.unique(
                sorted_log_times, return_index=True
            )
            if len(unique_log_times) < 2:  # Need at least two unique time points
                continue
            unique_log_losses = sorted_log_losses[unique_indices]

            # Perform interpolation in log-log scale
            log_interp_values = np.interp(
                np.log(time_grid), unique_log_times, unique_log_losses
            )
            interp_test = np.exp(log_interp_values)
            interp_test_losses_name.append(interp_test)

        # Store all interpolated curves for the method
        if interp_test_losses_name:
            all_interp_test_losses[name] = np.array(interp_test_losses_name)

        # Calculate median and quantiles across seeds
        if name in all_interp_test_losses and all_interp_test_losses[name].shape[0] > 0:
            median_curve = np.median(all_interp_test_losses[name], axis=0)
            q02_curve = np.quantile(all_interp_test_losses[name], 0.2, axis=0)
            q8_curve = np.quantile(all_interp_test_losses[name], 0.8, axis=0)

            # Plot median curve and shaded quantile region
            (line,) = ax_test.plot(time_grid, median_curve, label=name)
            color = line.get_color()
            ax_test.fill_between(
                time_grid, q02_curve, q8_curve, color=color, alpha=ALPHA_FILL
            )
            plotted_names_test.append(name)  # Mark name as plotted

    if not plotted_names_test:
        print("Warning: No data plotted for Test Loss.")
    else:
        ax_test.set_xscale("log")
        ax_test.set_yscale("log")
        ax_test.set_xlabel("Time (s)")
        ax_test.set_ylabel("Test Loss")
        ax_test.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir_1d / "corrected_interp1d_test_loss_true.pdf", format="pdf"
        )
    plt.close()

    # ============================        2D        ===============================
    # =============================================================================
    # Data Loading
    # =============================================================================
    results_dir_2d = Path(args.time_2d)
    if args.separate_2d:
        all_dfs_2d = []
        file_pattern_2d = results_dir_2d / "df_results_*.csv"
        csv_files_2d = list(results_dir_2d.glob("df_results_*.csv"))

        if not csv_files_2d:
            raise FileNotFoundError(
                f"No files found matching pattern: {file_pattern_2d}"
            )

        print(f"Found {len(csv_files_2d)} separate 2D result files:")
        for filepath in csv_files_2d:
            print(f"  - Loading {filepath.name}")
            try:
                # Extract method name from filename: df_results_['methodname'].csv
                method_name = (
                    filepath.stem.split("_")[-1] if "_" in filepath.stem else None
                )
                method_name = method_name.strip("[]'") if method_name else None

                if not method_name:
                    print(
                        "Warning: Could not extract method name from "
                        f"{filepath.name}. Skipping."
                    )
                    continue

                # If specific solvers are requested, only load those files
                if args.solver_2d is not None and method_name not in args.solver_2d:
                    print(f"    Skipping {filepath.name} (not in requested solvers).")
                    continue

                df_method = pd.read_csv(filepath)
                df_method["name"] = method_name
                all_dfs_2d.append(df_method)
            except pd.errors.EmptyDataError as e:
                print(f"    Error loading or processing {filepath.name}: {e}")
            except FileNotFoundError as e:
                print(f"    Error loading or processing {filepath.name}: {e}")

        if not all_dfs_2d:
            raise ValueError(
                "No valid 2D data loaded from separate CSV files (check filters?)."
            )

        df_results_2d = pd.concat(all_dfs_2d, ignore_index=True)
        print("Successfully combined 2D data from separate files.")
    else:
        single_csv_path_2d = results_dir_2d / "df_results.csv"
        if not single_csv_path_2d.is_file():
            raise FileNotFoundError(
                f"Single 2D results file not found: {single_csv_path_2d}"
            )
        print(f"Loading 2D results from single file: {single_csv_path_2d}")
        df_results_2d = pd.read_csv(single_csv_path_2d)

        # Filter based on --solver-2d argument if provided
        if args.solver_2d is not None:
            print(f"Filtering 2D results for solvers: {', '.join(args.solver_2d)}")
            original_rows_2d = len(df_results_2d)
            df_results_2d = df_results_2d[df_results_2d["name"].isin(args.solver_2d)]
            print(f"  Kept {len(df_results_2d)} out of {original_rows_2d} rows.")
            if df_results_2d.empty:
                print("No 2D data remaining after filtering for specified solvers.")

    # =============================================================================
    # Data Preprocessing
    # =============================================================================
    if not df_results_2d.empty:
        if "alphacsc" in df_results_2d["name"].unique():
            df_results_2d.loc[
                (df_results_2d["name"] == "alphacsc") & (df_results_2d["epoch"] == 0),
                "time",
            ] = 1e-1

            epoch0alphacsc = df_results_2d[
                (df_results_2d["name"] == "alphacsc") & (df_results_2d["epoch"] == 0)
            ].copy()

            all_dfs_preproc = [df_results_2d]
            original_names = df_results_2d["name"].unique()
            for name in original_names:
                if name == "alphacsc":
                    continue
                epoch0_copy = epoch0alphacsc.copy()
                epoch0_copy["name"] = name
                df_results_2d.loc[df_results_2d["name"] == name, "epoch"] += 1
                all_dfs_preproc.append(epoch0_copy)

            df_results_2d = pd.concat(all_dfs_preproc, ignore_index=True)
        else:
            print("alphacsc not found in the data, skipping epoch 0 adjustment.")

    # =============================================================================
    #   Plotting RUNTIME 2D
    # =============================================================================

    if df_results_2d.empty:
        print("Warning: No data to plot after filtering.")
        sys.exit()

    # Ensure data is sorted for cumulative sum and interpolation
    df_results_2d = df_results_2d.sort_values(by=["name", "seed", "epoch"])

    # Precompute cumulative time and adjusted losses
    # Check if 'time' column exists and has data before cumsum
    if "time" not in df_results_2d.columns or df_results_2d["time"].isna().all():
        print(
            "Warning: 'time' column missing or empty. Cannot compute cumulative time."
        )
        sys.exit()

    df_results_2d["cum_time"] = df_results_2d.groupby(["name", "seed"])["time"].cumsum()

    # Check if loss columns exist before proceeding
    if (
        "loss_true" not in df_results_2d.columns
        or "test_loss_true" not in df_results_2d.columns
    ):
        print(
            "Warning: 'loss_true' or 'test_loss_true' columns missing. "
            "Cannot plot losses."
        )
        sys.exit()

    min_loss_true = df_results_2d["loss_true"].min()
    min_test_loss_true = df_results_2d["test_loss_true"].min()
    # Add a small epsilon to avoid issues with log scale if min loss is 0
    epsilon = 1e-1
    df_results_2d["loss_true_adj"] = (
        df_results_2d["loss_true"] - min_loss_true + epsilon
    )
    df_results_2d["test_loss_true_adj"] = (
        df_results_2d["test_loss_true"] - min_test_loss_true + epsilon
    )

    # Determine global time range and create a common logarithmic time grid
    # Ensure min_time is strictly positive for logspace
    valid_times = df_results_2d[df_results_2d["cum_time"] > 0]["cum_time"]
    if valid_times.empty:
        print(
            "Warning: No positive cumulative time values found. "
            "Cannot create time grid."
        )
        sys.exit()
    min_time = max(valid_times.min(), 1e-4)
    max_time = df_results_2d["cum_time"].max()
    if min_time >= max_time:
        print(
            "Warning: Invalid time range for interpolation "
            f"({min_time=}, {max_time=}). Skipping plots."
        )
        sys.exit()
    n_grid_points = 500  # Number of points for interpolation grid
    time_grid = np.logspace(np.log10(min_time), np.log10(max_time), n_grid_points)

    # =============================================================================
    # Plotting Train Loss
    # =============================================================================
    plt.figure(figsize=(3.5, 2.2))
    ax_train = plt.gca()
    all_interp_train_losses_2d = {}
    plotted_names_train = []  # Keep track of names plotted

    for name in df_results_2d.name.unique():
        interp_train_losses_name = []
        df_name = df_results_2d[df_results_2d.name == name]
        for seed in df_name.seed.unique():
            df_run = df_name[df_name.seed == seed].copy()
            # Ensure time is monotonically increasing for interpolation
            df_run = df_run.drop_duplicates(subset=["cum_time"], keep="first")
            run_cum_time = df_run["cum_time"].to_numpy()
            run_loss_true = df_run["loss_true_adj"].to_numpy()

            # Skip runs with insufficient points for interpolation
            if len(run_cum_time) < 2:
                continue

            # Ensure data is valid for log transformation and interpolation
            valid_mask = (run_cum_time > 0) & (run_loss_true > 0)
            if np.sum(valid_mask) < 2:  # Need at least two points for interpolation
                continue

            current_times = run_cum_time[valid_mask]
            current_losses = run_loss_true[valid_mask]

            # Prepare data for log-log interpolation
            log_current_times = np.log(current_times)
            log_current_losses = np.log(current_losses)

            # Sort by log_current_times as np.interp requires xp to be increasing
            sort_indices = np.argsort(log_current_times)
            sorted_log_times = log_current_times[sort_indices]
            sorted_log_losses = log_current_losses[sort_indices]

            unique_log_times, unique_indices = np.unique(
                sorted_log_times, return_index=True
            )
            if len(unique_log_times) < 2:
                continue
            unique_log_losses = sorted_log_losses[unique_indices]

            # Perform interpolation in log-log scale
            # np.log(time_grid) are the target x-points (log of actual time values
            # on grid)
            # unique_log_times are the data x-points (log scale, sorted, unique)
            # unique_log_losses are the data y-points (log scale,
            # corresponding to unique_log_times)
            log_interp_values = np.interp(
                np.log(time_grid), unique_log_times, unique_log_losses
            )
            interp_train = np.exp(log_interp_values)
            interp_train_losses_name.append(interp_train)

        # Store all interpolated curves for the method
        if interp_train_losses_name:
            all_interp_train_losses_2d[name] = np.array(interp_train_losses_name)

        # Calculate median and quantiles across seeds
        if (
            name in all_interp_train_losses_2d
            and all_interp_train_losses_2d[name].shape[0] > 0
        ):
            median_curve = np.median(all_interp_train_losses_2d[name], axis=0)
            q02_curve = np.quantile(all_interp_train_losses_2d[name], 0.2, axis=0)
            q8_curve = np.quantile(all_interp_train_losses_2d[name], 0.8, axis=0)

            # Plot median curve and shaded quantile region
            (line,) = ax_train.plot(time_grid, median_curve, label=name)
            color = line.get_color()
            ax_train.fill_between(
                time_grid, q02_curve, q8_curve, color=color, alpha=ALPHA_FILL
            )
            plotted_names_train.append(name)

    if not plotted_names_train:
        print("Warning: No data plotted for Train Loss.")
    else:
        ax_train.set_xscale("log")
        ax_train.set_yscale("log")
        ax_train.set_xlabel("Time (s)")
        ax_train.set_ylabel("Train Loss")
        ax_train.legend()
        plt.tight_layout()
        plt.savefig(output_dir_2d / "corrected_interp1d_loss_true.pdf", format="pdf")
    plt.close()

    # =============================================================================
    # Plotting Test Loss
    # =============================================================================
    plt.figure(figsize=(3.5, 2.2))
    ax_test = plt.gca()
    all_interp_test_losses_2d = {}
    plotted_names_test = []

    for name in df_results_2d.name.unique():
        interp_test_losses_name = []
        df_name = df_results_2d[df_results_2d.name == name]
        for seed in df_name.seed.unique():
            df_run = df_name[df_name.seed == seed].copy()
            # Ensure time is monotonically increasing for interpolation
            df_run = df_run.drop_duplicates(subset=["cum_time"], keep="first")
            run_cum_time = df_run["cum_time"].to_numpy()
            run_test_loss_true = df_run["test_loss_true_adj"].to_numpy()

            # Skip runs with insufficient points for interpolation
            if len(run_cum_time) < 2:
                continue

            # Ensure data is valid for log transformation and interpolation
            valid_mask = (run_cum_time > 0) & (run_test_loss_true > 0)
            if np.sum(valid_mask) < 2:  # Need at least two points for interpolation
                continue

            current_times = run_cum_time[valid_mask]
            current_losses = run_test_loss_true[valid_mask]

            # Prepare data for log-log interpolation
            log_current_times = np.log(current_times)
            log_current_losses = np.log(current_losses)

            # Sort by log_current_times as np.interp requires xp to be increasing
            sort_indices = np.argsort(log_current_times)
            sorted_log_times = log_current_times[sort_indices]
            sorted_log_losses = log_current_losses[sort_indices]

            # Ensure uniqueness of sorted_log_times for np.interp
            unique_log_times, unique_indices = np.unique(
                sorted_log_times, return_index=True
            )
            if len(unique_log_times) < 2:  # Need at least two unique time points
                continue
            unique_log_losses = sorted_log_losses[unique_indices]

            # Perform interpolation in log-log scale
            log_interp_values = np.interp(
                np.log(time_grid), unique_log_times, unique_log_losses
            )
            interp_test = np.exp(log_interp_values)
            interp_test_losses_name.append(interp_test)

        # Store all interpolated curves for the method
        if interp_test_losses_name:
            all_interp_test_losses_2d[name] = np.array(interp_test_losses_name)

        # Calculate median and quantiles across seeds
        if (
            name in all_interp_test_losses_2d
            and all_interp_test_losses_2d[name].shape[0] > 0
        ):
            median_curve = np.median(all_interp_test_losses_2d[name], axis=0)
            q02_curve = np.quantile(all_interp_test_losses_2d[name], 0.2, axis=0)
            q8_curve = np.quantile(all_interp_test_losses_2d[name], 0.8, axis=0)

            # Plot median curve and shaded quantile region
            (line,) = ax_test.plot(time_grid, median_curve, label=name)
            color = line.get_color()
            ax_test.fill_between(
                time_grid, q02_curve, q8_curve, color=color, alpha=ALPHA_FILL
            )
            plotted_names_test.append(name)  # Mark name as plotted

    if not plotted_names_test:
        print("Warning: No data plotted for Test Loss.")
    else:
        ax_test.set_xscale("log")
        ax_test.set_yscale("log")
        ax_test.set_xlabel("Time (s)")
        ax_test.set_ylabel("Test Loss")
        ax_test.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir_2d / "corrected_interp1d_test_loss_true.pdf", format="pdf"
        )
    plt.close()

    # ============================    RECOVERY    ==================================
    # =============================================================================
    # Data Loading
    # =============================================================================
    df_recovery = pd.DataFrame()
    if args.recovery is not None:
        recovery_dir = Path(args.recovery)
        if args.separate_recovery:
            all_dfs = []
            file_pattern = recovery_dir / "df_results_*.csv"
            csv_files = list(recovery_dir.glob("df_results_*.csv"))

            if not csv_files:
                print(
                    f"No recovery files found matching pattern: {file_pattern}",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"Found {len(csv_files)} separate recovery files:")
            for filepath in csv_files:
                print(f"  - Loading {filepath.name}")
                try:
                    # Extract method name from filename: df_results_<name>.csv
                    method_name = (
                        filepath.stem.split("_")[-1] if "_" in filepath.stem else None
                    )
                    method_name = method_name.strip("[]'") if method_name else None

                    if not method_name:
                        print(
                            (
                                f"Warning: Could not extract method name from "
                                f"{filepath.name}. Skipping."
                            ),
                            file=sys.stderr,
                        )
                        continue

                    # If specific solvers are requested, only load those files
                    if (
                        args.solver_recover is not None
                        and method_name not in args.solver_recover
                    ):
                        print(
                            f"    Skipping {filepath.name} (not in requested solvers)."
                        )
                        continue

                    df_method = pd.read_csv(filepath)
                    df_method["name"] = method_name
                    all_dfs.append(df_method)
                except pd.errors.EmptyDataError as e:
                    print(
                        f"    Error loading or processing {filepath.name}: {e}",
                        file=sys.stderr,
                    )
                except FileNotFoundError as e:
                    print(
                        f"    Error loading or processing {filepath.name}: {e}",
                        file=sys.stderr,
                    )

            if not all_dfs:
                print("No recovery data loaded for specified solvers!", file=sys.stderr)
                sys.exit(1)
            df_recovery = pd.concat(all_dfs, ignore_index=True)
        else:
            # Load a single recovery file
            single_csv_path = (
                recovery_dir / "df_recovery.csv"
                if recovery_dir.is_dir()
                else Path(args.recovery)
            )
            if not single_csv_path.is_file():
                print(f"Recovery file not found: {single_csv_path}", file=sys.stderr)
                sys.exit(1)
            print(f"Loading recovery results from: {single_csv_path}")
            df_recovery = pd.read_csv(single_csv_path)

            # Filter based on --solver-recover argument if provided
            if args.solver_recover is not None:
                solvers = ", ".join(args.solver_recover)
                print(f"Filtering recovery results for solvers: {solvers}")
                original_rows = len(df_recovery)
                df_recovery = df_recovery[df_recovery["name"].isin(args.solver_recover)]
                print(f"  Kept {len(df_recovery)} out of {original_rows} rows.")
                if df_recovery.empty:
                    print(
                        "No recovery data remaining after filtering "
                        "for specified solvers.",
                        file=sys.stderr,
                    )
    # Check if we have any data after filtering
    if df_recovery.empty:
        print("No recovery data matches the specified criteria!", file=sys.stderr)
        sys.exit(1)

    print(f"Recovery score for solvers: {', '.join(df_recovery['name'].unique())}")

    # Plot recovery score
    fig, ax = plt.subplots(figsize=(4.2, 3.6))

    # Compute quantiles for recovery_score grouped by name and epoch
    quantiles = df_recovery.groupby(["name", "epoch"])["recovery_score"].quantile(
        [0.2, 0.5, 0.8]
    )
    quantiles = quantiles.reset_index().pivot_table(
        index=["name", "epoch"], columns="level_2", values="recovery_score"
    )  # Columns: 0.2, 0.5, 0.8

    plotted_names_recovery = []
    for name in df_recovery["name"].unique():
        if name not in quantiles.index.get_level_values("name"):
            continue
        epochs = quantiles.loc[name].index
        q20 = quantiles.loc[name][0.2]
        q50 = quantiles.loc[name][0.5]
        q80 = quantiles.loc[name][0.8]
        ax.fill_between(epochs, q20, q80, alpha=ALPHA_FILL)
        ax.plot(epochs, q50, label=name)
        plotted_names_recovery.append(name)

    if not plotted_names_recovery:
        print("Warning: No data plotted for Recovery Score.")
    else:
        print(f"Plotted recovery score for: {', '.join(plotted_names_recovery)}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recovery score")
    ax.legend()
    plt.tight_layout()
    plt.close(fig)

    # Interpolate recovery over time for each method/seed

    # Create combined figure with all plots
    if not df_results.empty and not df_results_2d.empty and args.recovery is not None:
        print("Creating combined figure with all plots...")
        plot_combined_figure(
            time_grid,
            all_interp_train_losses,
            all_interp_train_losses_2d,
            df_recovery,
            output_dir,
        )

        plot_combined_figure(
            time_grid,
            all_interp_test_losses,
            all_interp_test_losses_2d,
            df_recovery,
            output_dir,
            boxplot_recovery=True,
        )
