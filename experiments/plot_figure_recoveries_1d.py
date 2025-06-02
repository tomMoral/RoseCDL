import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        "lines.linewidth": 1,  # Linewidth for plot lines
        "pdf.fonttype": 42,  # Embed fonts in PDF for submission
        # "text.usetex": True,  # Enable LaTeX rendering for text
        "axes.labelpad": 1.8,  # Space between axis ticks and axis labels
        "xtick.major.pad": 1.5,  # Reduce space between x-tick and label
        "ytick.major.pad": 1.5,  # Reduce space between y-tick and label
    }
)


ALPHA_FILL = 0.1

# =============================================================================


def string_to_numpy(array_string):
    """Convert a string representation of a NumPy array to a NumPy array."""
    clean_string = array_string.replace("array(", "").replace(")", "")
    try:
        array_list = ast.literal_eval(clean_string)
        return np.array(array_list)
    except:
        try:
            return np.array(ast.literal_eval(array_string))
        except:
            print("Couldn't convert the string to a NumPy array")
            return None


def plot_line_recov_and_atoms(df_results, output_dir, min_reg, max_reg, atom_dict):
    fig = plt.figure(figsize=(6, 1.2))
    gs = plt.GridSpec(
        1,
        2,
        left=0.00,
        right=0.90,
        top=1,
        bottom=0.01,
        wspace=0.1,
    )

    # Create axes for the two main plots
    ax_recov = fig.add_subplot(gs[0, 0])

    gs_atoms = gs[0, 1].subgridspec(4, 2, hspace=0.7, wspace=0.05)
    ax_atoms = np.array(
        [[fig.add_subplot(gs_atoms[i, j]) for j in range(2)] for i in range(4)]
    )

    # Plot the recovery score
    df_results = df_results[
        (df_results["reg"] >= min_reg) & (df_results["reg"] <= max_reg)
    ]
    if df_results.empty:
        print(
            (
                f"No data found for regularization parameters in range "
                f"[{min_reg}, {max_reg}]!"
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    median = df_results.pivot_table(
        index="epoch", columns="reg", values="recovery_score", aggfunc="median"
    )
    q20 = df_results.pivot_table(
        index="epoch",
        columns="reg",
        values="recovery_score",
        aggfunc=lambda x: x.quantile(0.2),
    )
    q80 = df_results.pivot_table(
        index="epoch",
        columns="reg",
        values="recovery_score",
        aggfunc=lambda x: x.quantile(0.8),
    )
    median = median.clip(upper=1)
    q20 = q20.clip(upper=0.99)
    q80 = q80.clip(upper=0.99)
    regs_sorted = sorted(df_results["reg"].unique())
    cmap = plt.get_cmap("viridis", len(regs_sorted))
    palette_regs = [cmap(i) for i in range(len(regs_sorted))]
    reg_color_map = {reg: palette_regs[::-1][i] for i, reg in enumerate(regs_sorted)}

    for reg in regs_sorted:
        color = reg_color_map[reg]
        ax_recov.plot(
            median.index,
            median[reg],
            marker="o",
            markersize=2,
            label=f"{reg:.2f}",
            color=color,
        )
        ax_recov.fill_between(
            median.index,
            q20[reg],
            q80[reg],
            alpha=ALPHA_FILL,
            color=color,
            linewidth=0,
        )
    ax_recov.set_xlabel("Epoch")
    ax_recov.set_ylabel("Recovery Score")
    ax_recov.set_xlim(0, max(median.index))

    # Add legend only to the first plot (ax_recov)
    handles, labels = ax_recov.get_legend_handles_labels()
    handles = [
        Line2D(
            [0], [0], color=h.get_color(), linestyle="-", linewidth=h.get_linewidth()
        )
        for h in handles
    ]
    # Remove "\lambda=" from each legend entry, just show the value
    ax_recov.legend(
        handles,
        labels,
        loc="best",
        fontsize=7,
        ncol=1,
        handlelength=1,
        title=r"$\lambda$",
        title_fontsize=8,
    )

    true_atoms = string_to_numpy(df_results["true_dict"].reset_index(drop=True).iloc[0])
    for j in range(2):
        ax_atoms[0, j].plot(
            true_atoms[j][0],
            color=plt.get_cmap("tab10")(0),
        )
        ax_atoms[0, j].set_xticks([])
        ax_atoms[0, j].set_yticks([])

    row0_title_x = (
        ax_atoms[0, 0].get_position().x0 + ax_atoms[0, 1].get_position().x1
    ) / 2
    row0_title_y = ax_atoms[0, 0].get_position().y1 + 0.02
    fig.text(
        row0_title_x,
        row0_title_y,
        "True atoms",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # Plot the atoms
    for i, atom in enumerate(atom_dict, 1):
        reg = atom["reg"]
        seed = atom["seed"]
        atom_idx = atom["atom_idx"]
        df_atom = df_results[
            (df_results["reg"].round(2) == round(reg, 2))
            & (df_results["seed"].astype(str).str[:4] == str(seed)[:4])
            & (df_results["epoch"] == max(median.index))
        ]
        if df_atom.empty:
            print(
                (f"No data found for regularization parameter {reg} and seed {seed}!"),
                file=sys.stderr,
            )
            sys.exit(1)

        ax_atoms[i, 0].plot(
            string_to_numpy(df_atom["model_dict"].reset_index(drop=True).iloc[0])[
                atom_idx[0]
            ][0],
            color=plt.get_cmap("tab10")(0),
        )
        ax_atoms[i, 1].plot(
            string_to_numpy(df_atom["model_dict"].reset_index(drop=True).iloc[0])[
                atom_idx[1]
            ][0],
            color=plt.get_cmap("tab10")(0),
        )

        for j in range(2):
            ax_atoms[i, j].set_xticks([])
            ax_atoms[i, j].set_yticks([])

        rowi_title_x = (
            ax_atoms[i, 0].get_position().x0 + ax_atoms[i, 1].get_position().x1
        ) / 2
        rowi_title_y = ax_atoms[i, 0].get_position().y1 + 0.0
        fig.text(
            rowi_title_x,
            rowi_title_y,
            f"$\\lambda={reg:.2f}$",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.show()
    plt.savefig(
        output_dir / "plot_recovery_vs_reg_epochs.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )
    plt.close()


def plot_recovery_color_legend(
    df_results,
    output_dir,
    min_reg,
    max_reg,
):
    # Filter based on regularization parameter
    df_results = df_results[
        (df_results["reg"] >= min_reg) & (df_results["reg"] <= max_reg)
    ]
    if df_results.empty:
        print(
            (
                f"No data found for regularization parameters in range "
                f"[{min_reg}, {max_reg}]!"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    # Prepare data for plotting
    median = df_results.pivot_table(
        index="epoch", columns="reg", values="recovery_score", aggfunc="median"
    )
    q20 = df_results.pivot_table(
        index="epoch",
        columns="reg",
        values="recovery_score",
        aggfunc=lambda x: x.quantile(0.2),
    )
    q80 = df_results.pivot_table(
        index="epoch",
        columns="reg",
        values="recovery_score",
        aggfunc=lambda x: x.quantile(0.8),
    )

    regs_sorted = sorted(df_results["reg"].unique())
    cmap = plt.get_cmap("viridis", len(regs_sorted))
    palette_regs = [cmap(i) for i in range(len(regs_sorted))]
    reg_color_map = {reg: palette_regs[::-1][i] for i, reg in enumerate(regs_sorted)}

    # Create figure and axes
    fig = plt.figure(figsize=(5.5, 2))
    gs = plt.GridSpec(
        2,
        2,
        height_ratios=[0.1, 1],
        left=0.00,
        right=0.90,
        top=1,
        bottom=0.2,
        hspace=0.6,
        wspace=0.3,
    )
    ax_legend = fig.add_subplot(gs[0, :])
    ax_recov = fig.add_subplot(gs[1, 0])
    ax_legend.set_axis_off()
    gs_atoms = gs[1, 1].subgridspec(4, 1, hspace=0.5)
    ax_atoms = [fig.add_subplot(gs_atoms[i, 0]) for i in range(4)]

    # Plot the recovery score
    for reg in regs_sorted:
        color = reg_color_map[reg]
        ax_recov.plot(
            median.index,
            median[reg],
            marker="o",
            markersize=2,
            label=f"{reg:.2f}",
            color=color,
        )
        ax_recov.fill_between(
            median.index,
            q20[reg],
            q80[reg],
            alpha=ALPHA_FILL,
            color=color,
            linewidth=0,
        )

        # Add colorbar as legend for reg values with custom ticks at min, 0.5, max
        norm = mpl.colors.Normalize(vmin=min(regs_sorted), vmax=max(regs_sorted))
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=norm)
        sm.set_array([])
        # Set ticks at min, 0.5, max
        ticks = [min(regs_sorted), 0.5, max(regs_sorted)]
        cbar = fig.colorbar(
            sm,
            cax=ax_legend,
            orientation="horizontal",
            ticks=ticks,
        )
        cbar.set_label(r"$\lambda$ (regularization)", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
    ax_recov.set_xlabel("Epoch")
    ax_recov.set_ylabel("Recovery Score")
    ax_recov.set_xlim(0, max(median.index))

    # Add colorbar as legend for reg values
    norm = mpl.colors.Normalize(vmin=min(regs_sorted), vmax=max(regs_sorted))
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=ax_legend,
        orientation="horizontal",
        ticks=np.linspace(min(regs_sorted), max(regs_sorted), num=5),
    )
    cbar.set_label(r"$\lambda$ (regularization)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    plt.show()
    plt.savefig(
        output_dir / "plot_recovery_vs_reg_epochs_color.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    import argparse

    import matplotlib as mpl
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser(description="Plot recovery")
    parser.add_argument("results_dir", type=str, help="Input directory")
    parser.add_argument(
        "--output", "-o", type=str, help="Output directory", default="."
    )
    parser.add_argument(
        "--min-reg",
        type=float,
        help="Minimum regularization parameter to plot",
        default=0.0,
    )
    parser.add_argument(
        "--max-reg",
        type=float,
        help="Maximum regularization parameter to plot",
        default=1.0,
    )

    args = parser.parse_args()
    exp_dir = Path(args.results_dir)
    df_results = pd.read_csv(exp_dir / "results.csv")
    atom_dict = [
        {"reg": 0.1, "seed": 3849, "atom_idx": [0, 1]},
        {"reg": 0.35, "seed": 1634, "atom_idx": [0, 1]},
        {"reg": 0.8, "seed": 1634, "atom_idx": [1, 2]},
    ]

    plot_line_recov_and_atoms(
        df_results=df_results,
        output_dir=exp_dir,
        min_reg=args.min_reg,
        max_reg=args.max_reg,
        atom_dict=atom_dict,
    )
    print(f"Saved plot to {exp_dir / 'plot_recovery_vs_reg_epochs.png'}")

    plot_recovery_color_legend(
        df_results=df_results,
        output_dir=exp_dir,
        min_reg=args.min_reg,
        max_reg=args.max_reg,
    )
    print(f"Saved plot to {exp_dir / 'plot_recovery_vs_reg_epochs_color.png'}")
