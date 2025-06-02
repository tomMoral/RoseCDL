"""Plot the F1 score evolution over epochs for Rare event detection methods."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot F1 score evolution from results CSV."
)
parser.add_argument(
    "csv_path", type=str, help="Path to the CSV file containing the results"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="results",
    help="Directory to save the output plot (default: results)",
)
args = parser.parse_args()

# Ensure output directory exists
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Configure plot style for consistency
plt.rcParams.update(
    {
        "font.size": 9,  # Base font size
        "axes.labelsize": 11,  # Font size for x and y labels
        "axes.titlesize": 11,  # Font size for the title
        "xtick.labelsize": 8,  # Font size for x-axis tick labels
        "ytick.labelsize": 8,  # Font size for y-axis tick labels
        "legend.fontsize": 8,  # Font size for legend
        "lines.linewidth": 2,  # Line width
        "pdf.fonttype": 42,  # Embed fonts in PDF for submission
        "text.usetex": True,  # Use LaTeX for text rendering
    }
)


# Read the results
df_results = pd.read_csv(args.csv_path)

# Calculate statistics for each name-timing combination
curves = (
    df_results.groupby(["name", "timing", "epoch"])["f1"]
    .quantile([0.2, 0.5, 0.8])
    .reset_index()
    .pivot_table(index=["name", "timing", "epoch"], columns="level_3", values="f1")
)

cmap = plt.get_cmap("tab10")
colors = {name: cmap(i) for i, name in enumerate(df_results["name"].unique())}
linestyles = {"during": "-", "after": "--"}

# Create a separate plot for each name
for name in df_results["name"].unique():
    fig, ax = plt.subplots(figsize=(4, 3.5))

    # Get data for this name
    name_data = curves.loc[name]

    for timing in ["during", "after"]:
        if timing in name_data.index.get_level_values("timing"):
            group = name_data.loc[timing]

            # Plot the shaded area between 20th and 80th percentiles
            ax.fill_between(
                group.index,
                group[0.2],
                group[0.8],
                alpha=0.3,
                color=colors[name],
                label=None,
            )
            # Plot the median line
            ax.plot(
                group.index,
                group[0.5],
                label=timing.capitalize(),
                color=colors[name],
                linestyle=linestyles[timing],
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, df_results["epoch"].max())
    ax.set_xticks(range(0, df_results["epoch"].max() + 1, 5))
    ax.legend()

    # Save individual plot
    plt.tight_layout()
    fig.savefig(output_dir / f"f1_score_{name.replace(' ', '_').lower()}.pdf")
    plt.close()



# Combined plot for all methods
fig, ax = plt.subplots(figsize=(3.5, 3))
for name in df_results["name"].unique():
    name_data = curves.loc[name]

    for timing in ["during", "after"]:
        if timing in name_data.index.get_level_values("timing"):
            group = name_data.loc[timing]

            # Plot the shaded area between 20th and 80th percentiles
            ax.fill_between(
                group.index,
                group[0.2],
                group[0.8],
                alpha=0.3,
                color=colors[name],
                label=None,
            )
            # Plot the median line
            # Only add label for the first timing to avoid duplicates in legend
            label = name.capitalize() if timing == "during" else None
            ax.plot(
                group.index,
                group[0.5],
                label=label,
                color=colors[name],
                linestyle=linestyles[timing],
            )
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, df_results["f1"].max()+0.1)
ax.set_xlim(0, df_results["epoch"].max())
ax.set_xticks(range(0, df_results["epoch"].max() + 1, 5))

# Build combined legend handles with shorter lines
legend_line_length = 1.5  # shorter than default
method_handles = []
for name in df_results["name"].unique():
    # Split name into method_name and alpha value
    # e.g., "Method (alpha=0.1)" -> ("Method", "0.1")
    if " (alpha=" in name:
        method_name, alpha_part = name.split(" (alpha=")
        alpha_value = alpha_part[:-2]
        label = f"{method_name.capitalize()} ($\\alpha={alpha_value}$)"
    else:
        label = name
    method_handles.append(
        Line2D([0], [0], color=colors[name], linestyle="-", lw=2, label=label)
    )
timing_handles = [
    Line2D([0], [0], color="black", linestyle="-", lw=2, label="During"),
    Line2D([0], [0], color="black", linestyle="--", lw=2, label="After"),
]
handles = method_handles + timing_handles
labels = [h.get_label() for h in handles]
ax.legend(
    handles,
    labels,
    title=r"Methods \& Timing",
    handlelength=legend_line_length,  # reduce line length in legend
    loc="lower left",
)

# Save combined plot
plt.tight_layout()
fig.savefig(output_dir / "f1_score_all_methods.pdf")
plt.close()
# Print completion message
print(f"Plots saved to {output_dir}")



# ========================
# Boxplots for precision, recall, and F1

# Prepare data for boxplots: one box per (name, timing), with three metrics per tick
grouped = df_results.groupby(["name", "timing"])

labels = []
precision_data = []
recall_data = []
f1_data = []

for (name, timing), group in grouped:
    labels.append(f"{name}\n{timing}")
    precision_data.append(group["precision"].values)
    recall_data.append(group["recall"].values)
    f1_data.append(group["f1"].values)

n = len(labels)
x = np.arange(n)

fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 6))

width = 0.2  # width of each box
positions_precision = x - width
positions_recall = x
positions_f1 = x + width

# Plot each metric's boxplot at shifted positions
bp1 = ax.boxplot(
    precision_data,
    positions=positions_precision,
    widths=width,
    patch_artist=True,
    boxprops={"facecolor": "lightblue"},
    medianprops={"color": "blue"},
    showfliers=False,
)
bp2 = ax.boxplot(
    recall_data,
    positions=positions_recall,
    widths=width,
    patch_artist=True,
    boxprops={"facecolor": "lightgreen"},
    medianprops={"color": "green"},
    showfliers=False,
)
bp3 = ax.boxplot(
    f1_data,
    positions=positions_f1,
    widths=width,
    patch_artist=True,
    boxprops={"facecolor": "lightcoral"},
    medianprops={"color": "red"},
    showfliers=False,
)

# Set x-ticks in the center of the three boxes
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")

ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Precision, Recall, and F1 by Method and Timing")

# Custom legend
handles = [
    Line2D([0], [0], color="blue", lw=2, label="Precision"),
    Line2D([0], [0], color="green", lw=2, label="Recall"),
    Line2D([0], [0], color="red", lw=2, label="F1"),
]
ax.legend(handles=handles, loc="upper right")

plt.tight_layout()
fig.savefig(output_dir / "precision_recall_f1_boxplots.pdf")
plt.close()

# New plot with GridSpec
fig = plt.figure(figsize=(4, 3.5))
gs = GridSpec(2, 1, height_ratios=[0.1, 1], hspace=0.1, top=0.95, bottom=0.14)

# Create legend in the top subplot
ax_legend = fig.add_subplot(gs[0])
ax_legend.axis("off")

# Create main plot in the bottom subplot
ax = fig.add_subplot(gs[1])

for name in df_results["name"].unique():
    name_data = curves.loc[name]

    for timing in ["during", "after"]:
        if timing in name_data.index.get_level_values("timing"):
            group = name_data.loc[timing]
            ax.fill_between(
                group.index,
                group[0.2],
                group[0.8],
                alpha=0.2,
                color=colors[name],
                label=None,
            )
            label = name.capitalize() if timing == "during" else None
            ax.plot(
                group.index,
                group[0.5],
                label=label,
                color=colors[name],
                linestyle=linestyles[timing],
            )

ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, df_results["f1"].max()+0.1)
ax.set_xlim(0, df_results["epoch"].max())
ax.set_xticks(range(0, df_results["epoch"].max() + 1, 5))
# Add timing legend on the main plot
timing_handles = [
    Line2D([0], [0], color="black", linestyle="-", lw=2, label="During"),
    Line2D([0], [0], color="black", linestyle="--", lw=2, label="After"),
]
ax.legend(
    handles=timing_handles,
    title="Timing",
    loc="upper left",
    handlelength=legend_line_length,
)

# Build legend with shorter lines
legend_line_length = 1
method_handles = []
for name in df_results["name"].unique():
    if " (alpha=" in name:
        method_name, alpha_part = name.split(" (alpha=")
        alpha_value = alpha_part[:-1]
        label = f"{method_name.capitalize()} ($\\alpha={alpha_value}$)"
    else:
        label = name
    method_handles.append(
        Line2D([0], [0], color=colors[name], linestyle="-", lw=2, label=label)
    )

handles = method_handles
labels = [h.get_label() for h in handles]

# Place legend in the top subplot
ax_legend.legend(
    handles,
    labels,
    title=r"Methods",
    handlelength=legend_line_length,
    loc="center",
    ncol=len(method_handles),
    columnspacing=0.5,  # Reduce the space between columns in the legend
)
fig.savefig(output_dir / "f1_score_all_methods_with_top_legend.pdf")
plt.close()
