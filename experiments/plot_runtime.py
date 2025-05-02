import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d  # Add interp1d import

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
    """Plot train and test losses for different methods after interpolating
       onto a common time grid using scipy.interpolate.interp1d.

    Args:
        df_results: DataFrame containing the results
        output_dir: Directory to save the plots

    """
    # Ensure data is sorted for cumulative sum and interpolation
    df_results = df_results.sort_values(by=["name", "seed", "epoch"])

    # Precompute cumulative time and adjusted losses
    df_results["cum_time"] = df_results.groupby(["name", "seed"])["time"].cumsum()
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
    min_time = max(df_results[df_results["cum_time"] > 0]["cum_time"].min(), 1e-4)
    max_time = df_results["cum_time"].max()
    n_grid_points = 200  # Number of points for interpolation grid
    time_grid = np.logspace(np.log10(min_time), np.log10(max_time), n_grid_points)

    # --- Plotting Train Loss ---
    plt.figure(figsize=(3.5, 2.2))
    ax_train = plt.gca()
    all_interp_train_losses = {}

    for name in df_results.name.unique():
        interp_train_losses_name = []
        df_name = df_results[df_results.name == name]
        for seed in df_name.seed.unique():
            df_run = df_name[df_name.seed == seed].copy()
            # Ensure time is monotonically increasing for interpolation
            df_run = df_run.drop_duplicates(subset=["cum_time"], keep="first")
            run_cum_time = df_run["cum_time"].values
            run_loss_true = df_run["loss_true_adj"].values

            # Skip runs with insufficient points for interpolation
            if len(run_cum_time) < 2:
                continue

            # Create interpolation function using interp1d
            # Use first and last values as fill values for points outside the run's time range
            f_train = interp1d(
                run_cum_time,
                run_loss_true,
                kind="linear",
                bounds_error=False,
                fill_value=(run_loss_true[0], run_loss_true[-1]),
            )

            # Interpolate onto the common time grid
            interp_train = f_train(time_grid)
            interp_train_losses_name.append(interp_train)

        # Store all interpolated curves for the method
        all_interp_train_losses[name] = np.array(interp_train_losses_name)

        # Calculate median and quantiles across seeds
        if (
            all_interp_train_losses[name].shape[0] > 0
        ):  # Check if there are runs for this name
            median_curve = np.median(all_interp_train_losses[name], axis=0)
            q02_curve = np.quantile(all_interp_train_losses[name], 0.2, axis=0)
            q8_curve = np.quantile(all_interp_train_losses[name], 0.8, axis=0)

            # Plot median curve and shaded quantile region
            (line,) = ax_train.plot(time_grid, median_curve, label=name)
            color = line.get_color()
            ax_train.fill_between(
                time_grid, q02_curve, q8_curve, color=color, alpha=0.2
            )

    ax_train.set_xscale("log")
    ax_train.set_yscale("log")
    ax_train.set_xlabel("Time (s)")
    ax_train.set_ylabel("Train Loss")
    ax_train.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "interp1d_loss_true.pdf", format="pdf")  # Updated filename
    plt.close()

    # --- Plotting Test Loss ---
    plt.figure(figsize=(3.5, 2.2))
    ax_test = plt.gca()
    all_interp_test_losses = {}

    for name in df_results.name.unique():
        interp_test_losses_name = []
        df_name = df_results[df_results.name == name]
        for seed in df_name.seed.unique():
            df_run = df_name[df_name.seed == seed].copy()
            # Ensure time is monotonically increasing for interpolation
            df_run = df_run.drop_duplicates(subset=["cum_time"], keep="first")
            run_cum_time = df_run["cum_time"].values
            run_test_loss_true = df_run["test_loss_true_adj"].values

            # Skip runs with insufficient points for interpolation
            if len(run_cum_time) < 2:
                continue

            # Create interpolation function using interp1d
            f_test = interp1d(
                run_cum_time,
                run_test_loss_true,
                kind="linear",
                bounds_error=False,
                fill_value=(run_test_loss_true[0], run_test_loss_true[-1]),
            )

            # Interpolate onto the common time grid
            interp_test = f_test(time_grid)
            interp_test_losses_name.append(interp_test)

        # Store all interpolated curves for the method
        all_interp_test_losses[name] = np.array(interp_test_losses_name)

        # Calculate median and quantiles across seeds
        if (
            all_interp_test_losses[name].shape[0] > 0
        ):  # Check if there are runs for this name
            median_curve = np.median(all_interp_test_losses[name], axis=0)
            q02_curve = np.quantile(all_interp_test_losses[name], 0.2, axis=0)
            q8_curve = np.quantile(all_interp_test_losses[name], 0.8, axis=0)

            # Plot median curve and shaded quantile region
            (line,) = ax_test.plot(time_grid, median_curve, label=name)
            color = line.get_color()
            ax_test.fill_between(time_grid, q02_curve, q8_curve, color=color, alpha=0.2)

    ax_test.set_xscale("log")
    ax_test.set_yscale("log")
    ax_test.set_xlabel("Time (s)")
    ax_test.set_ylabel(f"Test Loss")
    ax_test.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "interp1d_test_loss_true.pdf", format="pdf"
    )  # Updated filename
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
        help="If set, load results from separate df_results_<name>.csv files instead of a single df_results.csv",
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
                # Extract method name from filename, e.g., df_results_['methodname'].csv
                method_name = (
                    filepath.stem.split("_")[2] if "_" in filepath.stem else None
                )
                # This gives us "['methodname']"
                method_name = method_name.strip("[]'") if method_name else None

                if not method_name:
                    print(
                        f"Warning: Could not extract method name from {filepath.name}. Skipping."
                    )
                    continue
                df_method = pd.read_csv(filepath)
                df_method["name"] = method_name
                all_dfs.append(df_method)
            except Exception as e:
                print(f"    Error loading or processing {filepath.name}: {e}")

        if not all_dfs:
            raise ValueError("No valid data loaded from separate CSV files.")

        df_results = pd.concat(all_dfs, ignore_index=True)
        print("Successfully combined data from separate files.")
    else:
        single_csv_path = results_dir / "df_results.csv"
        if not single_csv_path.is_file():
            raise FileNotFoundError(f"Single results file not found: {single_csv_path}")
        print(f"Loading results from single file: {single_csv_path}")
        df_results = pd.read_csv(single_csv_path)

    # --- Data Preprocessing ---

    # Adjust time for alphacsc epoch 0
    df_results.loc[
        (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0), "time"
    ] = 1e-1

    # Add the first loss evaluation of alphacsc to the other methods for fair comparison start
    epoch0alphacsc = df_results[
        (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0)
    ].copy()  # Use copy to avoid SettingWithCopyWarning

    all_dfs = [df_results]
    original_names = df_results["name"].unique()
    for name in original_names:
        if name == "alphacsc":
            continue
        # Create copies for each other method
        epoch0_copy = epoch0alphacsc.copy()
        epoch0_copy["name"] = name
        # Adjust epoch for existing data of this method
        df_results.loc[df_results["name"] == name, "epoch"] += 1
        all_dfs.append(epoch0_copy)

    # Concatenate original data (with adjusted epochs) and the added epoch 0 data
    df_results = pd.concat(all_dfs, ignore_index=True)

    # --- Plotting ---
    plot_losses(df_results, results_dir)
