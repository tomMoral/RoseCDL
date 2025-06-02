import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global parameters for NeurIPS style plots
plt.rcParams.update(
    {
        "font.size": 9,  # Base font size
        "axes.labelsize": 9,  # Font size for x and y labels
        "axes.titlesize": 9,  # Font size for the title
        "xtick.labelsize": 8,  # Font size for x-axis tick labels
        "ytick.labelsize": 8,  # Font size for y-axis tick labels
        "legend.fontsize": 8,  # Font size for the legend
        "lines.linewidth": 1.0,  # Linewidth for plot lines
        "pdf.fonttype": 42,  # Embed fonts in PDF for submission
    }
)


def plot_losses(df_results: pd.DataFrame, output_dir: Path) -> None:
    """Plot train and test losses for different methods after interpolating.

    Args:
        df_results: DataFrame containing the results (potentially filtered)
        output_dir: Directory to save the plots

    """
    if df_results.empty:
        print("Warning: No data to plot after filtering.")
        return

    # Ensure data is sorted for cumulative sum and interpolation
    df_results = df_results.sort_values(by=["name", "seed", "epoch"])

    # Precompute cumulative time and adjusted losses
    # Check if 'time' column exists and has data before cumsum
    if "time" not in df_results.columns or df_results["time"].isna().all():
        print(
            "Warning: 'time' column missing or empty. Cannot compute cumulative time."
        )
        return

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
        return

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
            "Warning: No positive cumulative time values found. Cannot create time grid."
        )
        return
    min_time = max(valid_times.min(), 1e-4)
    max_time = df_results["cum_time"].max()
    if min_time >= max_time:
        print(
            "Warning: Invalid time range for interpolation "
            f"({min_time=}, {max_time=}). Skipping plots."
        )
        return
    n_grid_points = 2000  # Number of points for interpolation grid
    time_grid = np.logspace(np.log10(min_time), np.log10(max_time), n_grid_points)

    # --- Plotting Train Loss ---
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
                time_grid, q02_curve, q8_curve, color=color, alpha=0.2
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
        plt.savefig(output_dir / "corrected_interp1d_loss_true.pdf", format="pdf")
    plt.close()

    # --- Plotting Test Loss ---
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
            ax_test.fill_between(time_grid, q02_curve, q8_curve, color=color, alpha=0.2)
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
        plt.savefig(output_dir / "corrected_interp1d_test_loss_true.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot results from runtime comparison experiment"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing the results CSV file(s)",
    )
    parser.add_argument(
        "--separate-csvs",
        action="store_true",
        help="If set, load results from separate df_results_<name>.csv "
        "files instead of a single df_results.csv",
    )
    parser.add_argument(
        "--solver",
        type=str,
        nargs="+",  # Accept one or more arguments
        default=None,  # Default to None if not provided
        help=(
            "Specify one or more solver names to plot. "
            "If not provided, all solvers found will be plotted."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.separate_csvs:
        all_dfs = []
        file_pattern = results_dir / "df_results_*.csv"
        csv_files = list(results_dir.glob("df_results_*.csv"))

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
                if args.solver is not None and method_name not in args.solver:
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
        single_csv_path = results_dir / "df_results.csv"
        if not single_csv_path.is_file():
            raise FileNotFoundError(f"Single results file not found: {single_csv_path}")
        print(f"Loading results from single file: {single_csv_path}")
        df_results = pd.read_csv(single_csv_path)

        # Filter based on --solver argument if provided
        if args.solver is not None:
            print(f"Filtering results for solvers: {', '.join(args.solver)}")
            original_rows = len(df_results)
            df_results = df_results[df_results["name"].isin(args.solver)]
            print(f"  Kept {len(df_results)} out of {original_rows} rows.")
            if df_results.empty:
                print(
                    "Warning: No data remaining after filtering for specified solvers."
                )

    # --- Data Preprocessing ---
    if not df_results.empty:
        # Adjust time for alphacsc epoch 0 if alphacsc is in the data
        if "alphacsc" in df_results["name"].unique():
            df_results.loc[
                (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0), "time"
            ] = 1e-1

            # Add the first loss evaluation of alphacsc to other methods for same start
            epoch0alphacsc = df_results[
                (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0)
            ].copy()  # Use copy to avoid SettingWithCopyWarning

            all_dfs_preproc = [
                df_results
            ]  # Start with the current (potentially filtered) data
            original_names = df_results["name"].unique()
            for name in original_names:
                if name == "alphacsc":
                    continue
                # Create copies for each other method present in the filtered data
                epoch0_copy = epoch0alphacsc.copy()
                epoch0_copy["name"] = name
                # Adjust epoch for existing data of this method
                df_results.loc[df_results["name"] == name, "epoch"] += 1
                all_dfs_preproc.append(epoch0_copy)

            # Concatenate original data and the added epoch 0 data
            df_results = pd.concat(all_dfs_preproc, ignore_index=True)
        else:
            print("alphacsc not found in the data, skipping epoch 0 adjustment.")

    # --- Plotting ---
    plot_losses(df_results, results_dir)

    epoch0alphacsc = df_results[
        (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0)
    ]
    for name in df_results["name"].unique():
        if name == "alphacsc":
            continue
        epoch0alphacsc.loc[:, "name"] = name
        df_results.loc[df_results["name"] == name, "epoch"] += 1
        df_results = pd.concat([df_results, epoch0alphacsc], ignore_index=True)

    # Plotting train loss for the different methods
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["loss_true"] = (
            median_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["loss_true"] = (
            q02_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        q8_curve = name_data[["time", "loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["loss_true"] = (
            q8_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        (line,) = ax.plot(median_curve["time"], median_curve["loss_true"], label=name)
        color = line.get_color()
        ax.fill_between(
            median_curve["time"],
            q02_curve["loss_true"],
            q8_curve["loss_true"],
            color=color,
            alpha=0.2,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Train Loss")
    plt.title("Speed of convergence of the different methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "naive_loss_true.png")
    plt.show()

    # Test loss for the different methods
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "test_loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["test_loss_true"] = (
            median_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "test_loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["test_loss_true"] = (
            q02_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        q8_curve = name_data[["time", "test_loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["test_loss_true"] = (
            q8_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        (line,) = ax.plot(
            median_curve["time"], median_curve["test_loss_true"], label=name
        )
        color = line.get_color()
        ax.fill_between(
            median_curve["time"],
            q02_curve["test_loss_true"],
            q8_curve["test_loss_true"],
            color=color,
            alpha=0.2,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Loss")
    plt.title("Test performance of the different methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "naive_test_loss_true.png")
    plt.show()

    epoch0alphacsc = df_results[
        (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0)
    ]
    for name in df_results["name"].unique():
        if name == "alphacsc":
            continue
        epoch0alphacsc.loc[:, "name"] = name
        df_results.loc[df_results["name"] == name, "epoch"] += 1
        df_results = pd.concat([df_results, epoch0alphacsc], ignore_index=True)

    # Plotting train loss for the different methods
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["loss_true"] = (
            median_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["loss_true"] = (
            q02_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        q8_curve = name_data[["time", "loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["loss_true"] = (
            q8_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        (line,) = ax.plot(median_curve["time"], median_curve["loss_true"], label=name)
        color = line.get_color()
        ax.fill_between(
            median_curve["time"],
            q02_curve["loss_true"],
            q8_curve["loss_true"],
            color=color,
            alpha=0.2,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Train Loss")
    plt.title("Speed of convergence of the different methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "naive_loss_true.png")
    plt.show()

    # Test loss for the different methods
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "test_loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["test_loss_true"] = (
            median_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "test_loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["test_loss_true"] = (
            q02_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        q8_curve = name_data[["time", "test_loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["test_loss_true"] = (
            q8_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        (line,) = ax.plot(
            median_curve["time"], median_curve["test_loss_true"], label=name
        )
        color = line.get_color()
        ax.fill_between(
            median_curve["time"],
            q02_curve["test_loss_true"],
            q8_curve["test_loss_true"],
            color=color,
            alpha=0.2,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Loss")
    plt.title("Test performance of the different methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "naive_test_loss_true.png")
    plt.show()
