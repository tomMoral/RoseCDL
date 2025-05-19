import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# =============================================================================
# Set global parameters for NeurIPS style plots (copied from plot_fig1.py)
# =============================================================================
plt.rcParams.update(
    {
        "font.size": 9,  # Base font size
        "axes.labelsize": 8,  # Font size for x and y labels
        "axes.titlesize": 8,  # Font size for the title
        "xtick.labelsize": 8,  # Font size for x-axis tick labels
        "ytick.labelsize": 8,  # Font size for y-axis tick labels
        "legend.fontsize": 7,  # Font size for the legend
        "lines.linewidth": 1.5,  # Linewidth for plot lines
        "pdf.fonttype": 42,  # Embed fonts in PDF for submission
        "text.usetex": True,  # Enable LaTeX rendering for text
    }
)


def plot_aggregated_multi_freq_subplot(
    ax,
    main_df_original,
    target_method,
    target_alpha_val,
    MIN_FREQ,
    current_plot_generates_legend,
):
    """Plot aggregated multi-frequency scores on a given subplot axis.

    Filters main_df by target_method and target_alpha_val.
    Plots final recovery_score and exo_score vs. frequency,
    with hue by 'reg' and style by 'sample_window'.
    """
    main_df = main_df_original.copy()  # Work on a copy

    # Convert target_alpha_val to its string representation for matching 'alpha_str'
    # Ensures consistent formatting e.g., -1.0 -> "-1.0", 3.5 -> "3.5"
    target_alpha_str = f"{float(target_alpha_val):.1f}"

    # --- Data Filtering ---
    # Filter by method: handle 'none', if it implies NaN or a specific string
    if target_method == "none":
        method_filter = main_df["method"].str.lower() == "none"
    else:
        method_filter = main_df["method"] == target_method

    df_subset = main_df[method_filter & (main_df["alpha_str"] == target_alpha_str)]

    print(f"--- Subplot: Method: {target_method}, Alpha: {target_alpha_str} ---")
    print(f"Data points pre-MIN_FREQ filter for this method/alpha: {len(df_subset)}")

    if df_subset.empty:
        ax.text(
            0.5,
            0.5,
            f"No data for\\nMethod: {target_method}, Alpha: {target_alpha_str}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.set_title(
            "Method : "
            f"{str(target_method).capitalize()} "
            f"(Alpha: {target_alpha_str})",
            fontsize=8,
        )
        return

    df_plot = df_subset.copy()
    if MIN_FREQ is not None:
        df_plot = df_plot[df_plot["freq"] >= MIN_FREQ]
        print(f"Data points after MIN_FREQ filter: {len(df_plot)}")
        if df_plot.empty:
            ax.text(
                0.5,
                0.5,
                f"No data after MIN_FREQ filter\\nMethod: {target_method}, "
                f"Alpha: {target_alpha_str}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
            )
            ax.set_title(
                f"Method : {str(target_method).capitalize()} "
                f"(Alpha: {target_alpha_str})",
                fontsize=8,
            )
            return

    # --- Calculate Final Scores (Quantiles at Max Epoch per Reg/SW/Freq) ---
    # Iterate over unique combinations of 'reg' and 'sample_window' to find max_epoch
    # This ensures we get the true "final" score for each series defined by reg and sw
    grouping_cols_for_final_epoch = [
        "reg",
        "sample_window",
        "freq",
        "run_index",
    ]  # run_index to be safe

    # Get data from the last epoch for each group
    idx = df_plot.groupby(grouping_cols_for_final_epoch)["epoch"].idxmax()
    df_final_epoch_data = df_plot.loc[idx]
    print("df_final_epoch_data.info() after creation:")
    df_final_epoch_data.info(verbose=True, show_counts=True)

    # --- Diagnostic prints moved after df_final_epoch_data is defined ---
    print(f"Subplot ({target_method}, {target_alpha_str}):")
    if "recovery_score" in df_final_epoch_data.columns:
        print(f"  'recovery_score' found for {target_method}, {target_alpha_str}.")
    else:
        print(f"  WARN: No 'recovery_score' for {target_method}, {target_alpha_str}.")

    if "exo_score" in df_final_epoch_data.columns:
        print(f"  'exo_score' found for {target_method}, {target_alpha_str}.")
    else:
        print(f"  WARN: No 'exo_score' for {target_method}, {target_alpha_str}.")
    # --- End Diagnostic prints ---

    all_final_quantiles_list = []
    all_final_exo_quantiles_list = []

    # Calculate quantiles using pivot_table
    if (
        "recovery_score" in df_final_epoch_data.columns
        and not df_final_epoch_data.empty
    ):
        quantiles_rs = (
            df_final_epoch_data.groupby(["freq", "sample_window", "reg"])[
                "recovery_score"
            ]
            .quantile([0.2, 0.5, 0.8])
            .reset_index()
        )
        # Assuming the quantile level added by reset_index() is second to last column
        # and the value column ('recovery_score') is the last.
        if not quantiles_rs.empty:
            q = (
                quantiles_rs.pivot_table(
                    index=["freq", "sample_window", "reg"],
                    columns=quantiles_rs.columns[-2],
                    values="recovery_score",
                )
                .reset_index()
                .rename(columns={0.5: "median", 0.2: "q20", 0.8: "q80"})
            )
            all_final_quantiles_list.append(q)

    if "exo_score" in df_final_epoch_data.columns and not df_final_epoch_data.empty:
        quantiles_exo = (
            df_final_epoch_data.groupby(["freq", "sample_window", "reg"])["exo_score"]
            .quantile([0.2, 0.5, 0.8])
            .reset_index()
        )
        if not quantiles_exo.empty:
            q_exo = (
                quantiles_exo.pivot_table(
                    index=["freq", "sample_window", "reg"],
                    columns=quantiles_exo.columns[-2],
                    values="exo_score",
                )
                .reset_index()
                .rename(columns={0.5: "median", 0.2: "q20", 0.8: "q80"})
            )
            all_final_exo_quantiles_list.append(q_exo)

    if not all_final_quantiles_list:
        ax.text(
            0.5,
            0.5,
            (
                f"No final quantiles to plot\\nMethod: {target_method}, "
                f"Alpha: {target_alpha_str}"
            ),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.set_title(
            f"Method : {str(target_method).capitalize()} (Alpha: {target_alpha_str})",
            fontsize=8,
        )
        return

    agg_final_quantiles = pd.concat(all_final_quantiles_list, ignore_index=True)
    agg_has_exo = bool(all_final_exo_quantiles_list)
    agg_final_exo_quantiles = (
        pd.concat(all_final_exo_quantiles_list, ignore_index=True)
        if agg_has_exo
        else pd.DataFrame()
    )
    # More diagnostics after attempting to create agg_final_exo_quantiles
    if agg_has_exo:
        if agg_final_exo_quantiles.empty:
            print(
                f"  WARN: ({target_method}, {target_alpha_str}), "
                "exo_score data but no quantiles."
            )
        else:
            print(
                f"  ({target_method}, {target_alpha_str}), "
                "agg_final_exo_quantiles created."
            )
    else:
        print(f"  ({target_method}, {target_alpha_str}), no exo_scores to plot.")

    # --- Plotting ---
    # Define palettes and styles
    unique_regs_plot = sorted(agg_final_quantiles["reg"].unique())

    cmap = plt.get_cmap("viridis", len(unique_regs_plot))
    palette_regs = [cmap(i) for i in range(len(unique_regs_plot))]
    reg_color_map = {
        reg: palette_regs[::-1][i] for i, reg in enumerate(unique_regs_plot)
    }

    unique_sw_plot = sorted(agg_final_quantiles["sample_window"].unique())

    # Plot recovery_score
    # Keep track of regs for which a legend entry has been made for recovery scores
    legend_added_for_reg_rec = set()

    for reg_val in unique_regs_plot:
        for sw_val_orig in unique_sw_plot:  # sw_val_orig is the original type from df
            df_slice = agg_final_quantiles[
                (agg_final_quantiles["reg"] == reg_val)
                & (
                    agg_final_quantiles["sample_window"] == sw_val_orig
                )  # Filter with original type
            ]
            if not df_slice.empty:
                label_rec = None  # Legend handled globally
                if (
                    current_plot_generates_legend
                    and reg_val not in legend_added_for_reg_rec
                ):  # Add labels only for the first plot for the main legend
                    # Format reg_val to one decimal place if it's a float,
                    # else convert to string
                    try:
                        label_rec = f"$\\lambda$: {float(reg_val) / 7.23:.3f}"
                    except ValueError:
                        label_rec = f"$\\lambda$: {str(reg_val)!s}"
                    legend_added_for_reg_rec.add(reg_val)

                # Convert sns.lineplot to ax.plot for recovery score
                ax.plot(
                    df_slice["freq"],
                    df_slice["median"],
                    color=reg_color_map.get(reg_val),
                    linestyle="--",
                    marker="o",
                    markersize=4.3,
                    label=label_rec,
                )
                ax.fill_between(
                    df_slice["freq"],
                    df_slice["q20"],
                    df_slice["q80"],
                    color=reg_color_map.get(reg_val),
                    alpha=0.1,
                )

    # Plot exo_score if exists
    if agg_has_exo and not agg_final_exo_quantiles.empty:
        print(f"  Plotting exo_score for {target_method}, {target_alpha_str}.")
        # Use a different color palette or modify existing for exo scores if desired
        # For simplicity, using same reg colors but will rely on linestyle and label
        unique_regs_exo_plot = sorted(agg_final_exo_quantiles["reg"].unique())

        for reg_val in unique_regs_exo_plot:
            for sw_val_orig in sorted(
                agg_final_exo_quantiles["sample_window"].unique()
            ):
                df_exo_slice = agg_final_exo_quantiles[
                    (agg_final_exo_quantiles["reg"] == reg_val)
                    & (agg_final_exo_quantiles["sample_window"] == sw_val_orig)
                ]
                if not df_exo_slice.empty:
                    label_exo = None  # Exo scores do not get a legend entry

                    # Convert sns.lineplot to ax.plot for exo score
                    ax.plot(
                        df_exo_slice["freq"],
                        df_exo_slice["median"],
                        color=reg_color_map.get(reg_val),
                        linestyle="-",
                        marker="x",
                        markersize=4.3,
                        label=label_exo,
                    )
                    ax.fill_between(
                        df_exo_slice["freq"],
                        df_exo_slice["q20"],
                        df_exo_slice["q80"],
                        color=reg_color_map.get(reg_val),
                        alpha=0.1,
                    )

    ax.set_xlabel("Rare event frequency")
    ax.set_ylabel("Recovery Score")
    ax.set_xscale("log")
    # Set title for the subplot
    title_method_str = str(target_method)
    if title_method_str.lower() == "none":
        title_method_str = "No inline outlier detection"
        ax.set_title(title_method_str)
    else:
        title_method_str = title_method_str.capitalize()
        ax.set_title(f"Method : {title_method_str} (Alpha: {target_alpha_str})")


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


def plot_dictionary_quality(fig, ax):
    """Plot dictionary quality comparison in the given axis."""
    # Load the data files
    workspace_root = Path(__file__).resolve().parent.parent

    data_09 = pd.read_csv(
        workspace_root / "results/dict_quality/dict_quality_09/results_4263093181.csv"
    )
    dict_09 = string_to_numpy(data_09.query("epoch == 21")["model_dict"].to_numpy()[0])

    data_01 = pd.read_csv(
        workspace_root / "results/dict_quality/dict_quality_01/results_4008076976.csv"
    )
    dict_01 = string_to_numpy(data_01.query("epoch == 24")["model_dict"].to_numpy()[0])

    data_05 = pd.read_csv(
        workspace_root / "results/dict_quality/dict_quality_05/results_3440524841.csv"
    )
    dict_05 = string_to_numpy(data_05.query("epoch == 28")["model_dict"].to_numpy()[0])

    true_dict = np.load(
        workspace_root.parent
        / "2data/data/True_data/ROSE_5000/new_text_4_5000_ROSE_exo_Z_0p0620.npz"
    ).get("D")

    # Plot indices
    idx_01 = [1, 5, 3, 0]
    idx_05 = [0, 3, 4, 5]
    idx_09 = [1, 0, 4, 3]

    # Create 2x2 grid within the subplot
    gs_dict = ax.get_gridspec()
    # Shift the subgridspec to the left by adjusting left and right
    subgs = gs_dict[1, 2].subgridspec(4, 4, hspace=0.3, wspace=-0.7)

    # Clear the main axis
    ax.clear()
    ax.axis("off")  # Hide the main axis

    # Create new small axes
    axs = [[fig.add_subplot(subgs[i, j]) for j in range(4)] for i in range(4)]

    row_titles = [
        "True dictionary",
        "Dict. recovery: 0.54",
        "Dict. recovery: 0.87",
        "Dict. recovery: 0.93",
    ]

    for i in range(4):
        axs[0][i].imshow(true_dict[i], cmap="gray")
        axs[1][i].imshow(dict_01[idx_01[i], 0], cmap="gray")
        axs[2][i].imshow(dict_05[idx_05[i], 0], cmap="gray")
        axs[3][i].imshow(dict_09[idx_09[i], 0], cmap="gray")

        for j in range(4):
            axs[j][i].axis("off")

    # Add row titles
    for i, title in enumerate(row_titles):
        axs[i][0].annotate(
            title,
            xy=(2.22, 1.15),
            xycoords="axes fraction",
            va="center",
            ha="center",
            fontsize=8,
        )

    # I want to move this plot to the right
    # Adjust the position of the entire grid
    for i in range(4):
        for j in range(4):
            axs[i][j].set_position(
                [
                    axs[i][j].get_position().x0 - 0.05,
                    axs[i][j].get_position().y0,
                    axs[i][j].get_position().width,
                    axs[i][j].get_position().height,
                ]
            )


def plot_main_figure_fig2(main_df, output_dir_plots_root, MIN_FREQ_config):
    """Plot the figure for the multi-frequency comparison."""
    fig = plt.figure(figsize=(9, 3.8))  # Adjusted height for suptitle and legend
    gs = plt.GridSpec(
        2,
        3,
        height_ratios=[0.1, 1],  # Adjusted legend height ratio
        left=0.07,
        right=1.15,
        top=0.85,
        bottom=0.15,
        hspace=0.2,
        wspace=0.2,
    )

    ax_legend = fig.add_subplot(gs[0, :])
    ax_mult_freq_none = fig.add_subplot(gs[1, 0])
    ax_mult_freq_mad = fig.add_subplot(gs[1, 1])
    ax_third_plot = fig.add_subplot(gs[1, 2])

    ax_legend.set_axis_off()

    # Plot 1: (method='none', alpha=-1.0)
    plot_aggregated_multi_freq_subplot(
        ax_mult_freq_none,
        main_df,
        target_method="none",
        target_alpha_val=-1.0,
        MIN_FREQ=MIN_FREQ_config,
        current_plot_generates_legend=True,
    )

    # Plot 2: (method='mad', alpha=3.5)
    plot_aggregated_multi_freq_subplot(
        ax_mult_freq_mad,
        main_df,
        target_method="mad",
        target_alpha_val=3.5,
        MIN_FREQ=MIN_FREQ_config,
        current_plot_generates_legend=False,
    )

    # Plot 3: Dictionary quality comparison
    plot_dictionary_quality(fig, ax_third_plot)

    # --- Shared Legend ---
    handles, labels = [], []
    # Collect from the plot that was designated to generate legend items
    if ax_mult_freq_none.lines:  # Check if lines were plotted
        handles_plot1, labels_plot1 = ax_mult_freq_none.get_legend_handles_labels()
        # Make all legend handles solid lines regardless of original linestyle
        for h, label in zip(handles_plot1, labels_plot1, strict=False):
            color = h.get_color() if hasattr(h, "get_color") else "black"
            handles.append(
                Line2D(
                    [0], [0], color=color, linestyle="-", linewidth=h.get_linewidth()
                )
            )
            labels.append(label)

    # Add line type examples
    handles.extend(
        [
            Line2D([0], [0], color="black", linestyle="-"),
            Line2D([0], [0], color="black", linestyle="--"),
        ]
    )
    labels.extend(["Rare event dict Recovery", "True dict Recovery"])

    # Create a dictionary to preserve unique handles and labels, maintaining order
    unique_legend_items = dict(zip(labels, handles, strict=False))
    ax_legend.legend(
        unique_legend_items.values(),
        unique_legend_items.keys(),
        loc="center",
        ncol=len(unique_legend_items),
        fontsize=7,
    )

    # No suptitle for now
    # To add here if needed

    output_file = output_dir_plots_root / "plot_fig2_multi_freq_comparison.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    # Determine workspace root and paths assuming script is in experiments/
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    output_dir_main = workspace_root / "plots"
    fig2_output_dir = output_dir_main / "TEST"
    fig2_output_dir.mkdir(parents=True, exist_ok=True)

    # Path to the data file, relative to workspace_root
    data_file_path = (
        workspace_root
        / "results"
        / "final_multi_freq"
        / "df_results_multi_freq_reg_window.csv"
    )

    print(f"Attempting to load data from: {data_file_path}")

    main_df = None  # Initialize main_df
    try:
        main_df = pd.read_csv(data_file_path)
        print(f"Successfully loaded data. Shape: {main_df.shape}")
        print(f"Columns in loaded CSV: {main_df.columns.tolist()}")
        if "exo_score" not in main_df.columns:
            print("Warning: 'exo_score' column not found in the input CSV.")
            print(
                "Exo scores cannot be plotted if the column is missing from the source."
            )
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file_path}")
        print("Please ensure the CSV file exists at the expected location.")
        sys.exit(1)  # Exit if file not found
    except pd.errors.EmptyDataError:
        print(f"Error: Data file is empty at {data_file_path}.")
        sys.exit(1)  # Exit if file is empty
    except Exception as e:  # Catch other pandas/general exceptions during load
        print(f"Error loading or parsing CSV at {data_file_path}: {e}")
        sys.exit(1)  # Exit on other load errors
    if main_df is None or main_df.empty:
        print("DataFrame is None or empty after attempting to load. Exiting.")
        sys.exit(1)

    # --- Preprocessing (adapted from plot_multiple_frequencies.py) ---

    # Check for 'sample_window' AFTER main_df is confirmed to be loaded and not empty
    if "sample_window" not in main_df.columns:
        print(
            "Error: 'sample_window' column is required but not found in the DataFrame."
        )
        print("Cannot determine a default for 'sample_window'. Exiting.")
        sys.exit(1)

    if "method" not in main_df.columns:
        print("Warning: 'method' column not found. Assigning value 'default_method'.")
        main_df["method"] = "default_method"
    # Fill NaN methods with a string placeholder like "None" (string)
    # to distinguish from actual string "none" if used
    # This helps in filtering later if 'none' method means NaN in original data
    main_df["method"] = main_df["method"].fillna("None").astype(str)

    if "alpha" not in main_df.columns:
        print("Warning: 'alpha' column not found. Assigning default value 0.0.")
        main_df["alpha"] = 0.0

    # Create/format 'alpha_str' for consistent filtering
    # Handles numeric alphas (e.g., 3.5, -1.0) and string alphas
    main_df["alpha_str"] = (
        main_df["alpha"]
        .apply(
            lambda x: f"{float(x):.1f}"
            if pd.notna(x) and isinstance(x, float | int)
            else str(x)
        )
        .astype(str)
    )

    print("\nSample of preprocessed main_df head:")
    print(
        main_df[["method", "alpha", "alpha_str", "reg", "sample_window", "freq"]].head()
    )
    print(f"\nUnique methods found: {main_df['method'].unique()}")
    print(f"Unique alpha_str found: {main_df['alpha_str'].unique()}")

    MIN_FREQ_config = 1e-2

    plot_main_figure_fig2(main_df, fig2_output_dir, MIN_FREQ_config)
