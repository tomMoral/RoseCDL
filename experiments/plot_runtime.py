import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Set global parameters for NeurIPS style plots
plt.rcParams.update({
    "font.size": 9,          # Base font size
    "axes.labelsize": 9,     # Font size for x and y labels
    "axes.titlesize": 9,     # Font size for the title
    "xtick.labelsize": 8,    # Font size for x-axis tick labels
    "ytick.labelsize": 8,    # Font size for y-axis tick labels
    "legend.fontsize": 8,    # Font size for the legend
    "lines.linewidth": 1.0,  # Linewidth for plot lines
    "pdf.fonttype": 42,      # Embed fonts in PDF for submission
})

def plot_losses(df_results: pd.DataFrame, output_dir: Path) -> None:
    """Plot train and test losses for different methods.

    Args:
        df_results: DataFrame containing the results
        output_dir: Directory to save the plots
    """
    # Plotting train loss for the different methods
    plt.figure(figsize=(3.5, 2.2))
    ax = plt.gca()
    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["loss_true"] = median_curve["loss_true"] - df_results["loss_true"].min() + 1e1

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["loss_true"] = q02_curve["loss_true"] - df_results["loss_true"].min() + 1e1

        q8_curve = name_data[["time", "loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["loss_true"] = q8_curve["loss_true"] - df_results["loss_true"].min() + 1e1

        line, = ax.plot(median_curve["time"], median_curve["loss_true"], label=name)
        color = line.get_color()
        ax.fill_between(median_curve["time"], q02_curve["loss_true"], q8_curve["loss_true"], color=color, alpha=0.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "new_loss_true.pdf", format="pdf")
    plt.close()

    # Test loss for the different methods
    plt.figure(figsize=(3.5, 2.2))
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

        line, = ax.plot(
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
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "new_test_loss_true.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot results from runtime comparison experiment"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing the results (df_results.csv)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df_results = pd.read_csv(results_dir / "df_results.csv")

    df_results.loc[(df_results["name"] == "alphacsc") & (df_results["epoch"] == 0), 'time'] *= 10

    # Adding the first loss evaluation of alphacsc to the other methods
    epoch0alphacsc = df_results[
        (df_results["name"] == "alphacsc") & (df_results["epoch"] == 0)
    ]
    for name in df_results["name"].unique():
        if name == "alphacsc":
            continue
        epoch0alphacsc.loc[:, "name"] = name
        df_results.loc[df_results["name"] == name, "epoch"] += 1
        df_results = pd.concat([df_results, epoch0alphacsc], ignore_index=True)

    plot_losses(df_results, results_dir)
