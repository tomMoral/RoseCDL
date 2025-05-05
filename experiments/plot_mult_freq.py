# %%
from pathlib import Path

import matplotlib.lines as mlines  # For custom legends
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
# --- Configuration ---
PLOT_PER_WINDOW = True  # Set to True to generate one plot per window size,
# False for one plot with all windows
MIN_FREQ = (
    1e-2  # Set to a float value (e.g., 0.001) to plot only frequencies >= this value.
)
# Set to None to plot all.
# --- End Configuration ---

main_df = pd.read_csv(
    "../results/multi_window_freq_exp_OD/df_results_multi_freq_reg_window.csv"
)
exp_dir = Path("../results/multi_window_freq_exp_OD/")
# %%
main_df.head()
# %%
# Calculate n_runs if 'run_index' exists
if "run_index" in main_df.columns:
    n_runs = main_df["run_index"].nunique()
else:
    n_runs = 1  # Assume 1 run if no run_index column
    print("Warning: 'run_index' column not found. Assuming n_runs=1.")
    main_df["run_index"] = 0  # Add a dummy run_index if needed for grouping

# Add 'reg' column if it doesn't exist (assuming the filename implies reg=0.8)
# This might need adjustment if the CSV contains multiple reg values already
if "reg" not in main_df.columns:
    print("Warning: 'reg' column not found. Assuming reg=0.8 based on filename.")
    main_df["reg"] = 0.8

# Ensure 'sample_window' exists
if "sample_window" not in main_df.columns:
    print("Error: 'sample_window' column is required but not found.")
    # Handle error appropriately, e.g., exit or assign a default
    # For now, let's assume a default if it's missing, though this is risky
    main_df["sample_window"] = (
        main_df["sample_window"].unique()[0]
        if main_df["sample_window"].nunique() > 0
        else "default_ws"
    )
    print(
        "Warning: Assigning default sample_window: "
        f"{main_df['sample_window'].unique()[0]}"
    )

# --- Check for method and alpha columns ---
if "method" not in main_df.columns:
    print(
        "Warning: 'method' column not found. Assigning default value 'default_method'."
    )
    main_df["method"] = "default_method"
if "alpha" not in main_df.columns:
    print("Warning: 'alpha' column not found. Assigning default value 0.0.")
    main_df["alpha"] = 0.0
# --- End Check ---

# %%
# --- Get unique method and alpha combinations ---
method_alpha_cols = ["method", "alpha"]
if not all(col in main_df.columns for col in method_alpha_cols):
    print(f"Error: Required columns {method_alpha_cols} not found in DataFrame.")
    # Exit or handle error appropriately
    unique_method_alpha_pairs = []
else:
    # Convert alpha to string for consistent grouping/filenames if it's numeric
    if pd.api.types.is_numeric_dtype(main_df["alpha"]):
        main_df["alpha_str"] = main_df["alpha"].astype(str)
        method_alpha_cols = ["method", "alpha_str"]  # Use string version for pairs

    unique_method_alpha_pairs = (
        main_df[method_alpha_cols].drop_duplicates().to_numpy().tolist()
    )
    print(f"Found {len(unique_method_alpha_pairs)} unique (method, alpha) pairs.")
# --- End Get unique combinations ---


# Plotting section - loop over unique (method, alpha) pairs
print(f"Generating plots for (method, alpha) pairs: {unique_method_alpha_pairs}")

for method_val, alpha_val_str in unique_method_alpha_pairs:  # Use alpha_str here
    # Try to convert alpha_val_str back to numeric for potential internal use if needed
    try:
        alpha_val = pd.to_numeric(alpha_val_str)
    except ValueError:
        alpha_val = alpha_val_str  # Keep as string if conversion fails

    # Initialize lists to store quantiles for the (method, alpha) pair for the agg. plot
    all_final_quantiles = []
    all_final_exo_quantiles = []
    processed_regs = []  # Keep track of regs that had data within this method/alpha

    # Filter main DataFrame for the current method and alpha
    df_subset = main_df[
        (main_df["method"] == method_val)
        & (main_df[method_alpha_cols[1]] == alpha_val_str)
    ].copy()  # Use alpha_str for filtering

    print(f"\n--- Processing Method: {method_val}, Alpha: {alpha_val_str} ---")
    print(f"Data points for this combination: {len(df_subset)}")

    if df_subset.empty:
        print(
            f"Skipping plots for Method={method_val}, "
            f"Alpha={alpha_val_str} as no data is available."
        )
        continue

    # --- Apply Frequency Filter ---
    df_plot = (
        df_subset.copy()
    )  # Use df_plot for actual plotting within this method/alpha
    if MIN_FREQ is not None:
        print(f"Filtering data for freq >= {MIN_FREQ}")
        df_plot = df_plot[df_plot["freq"] >= MIN_FREQ]
        if df_plot.empty:
            print(
                f"Skipping plots for Method={method_val}, "
                f"Alpha={alpha_val_str} after frequency filtering "
                f"(freq >= {MIN_FREQ})."
            )
            continue
        print(f"Data points after frequency filtering: {len(df_plot)}")
    # --- End Frequency Filter ---

    # --- Plot final recovery score AND exo score vs frequency
    # (per reg within method/alpha) ---
    # This section now generates plots for each reg value found
    # *within* the current (method, alpha) subset
    unique_regs_in_subset = sorted(df_plot["reg"].unique())
    print(f"Found reg values within this subset: {unique_regs_in_subset}")

    for reg_val in unique_regs_in_subset:
        df_plot_reg = df_plot[
            df_plot["reg"] == reg_val
        ].copy()  # Further filter by reg for these plots
        print(
            "Plotting for reg = "
            f"{reg_val} (within Method={method_val}, Alpha={alpha_val_str}) "
            f"({len(df_plot_reg)} data points)"
        )

        if df_plot_reg.empty:
            print(
                f"Skipping plots for reg={reg_val} as no data is "
                f"available for this specific reg."
            )
            continue

        if "recovery_score" in df_plot_reg.columns:
            # Group by freq, run_idx, and sw to get the last epoch for each combination
            final_scores_idx = df_plot_reg.groupby(
                ["freq", "run_index", "sample_window"]
            )["epoch"].idxmax()
            final_scores = df_plot_reg.loc[final_scores_idx]

            # Quantiles for the final recovery score per freq and sample_window
            final_quantiles = (
                final_scores.groupby(["freq", "sample_window"])["recovery_score"]
                .quantile([0.2, 0.5, 0.8])
                .unstack()
                .reset_index()
            )
            final_quantiles = final_quantiles.rename(
                columns={0.5: "median", 0.2: "q20", 0.8: "q80"}
            )

            # Check and calculate quantiles for the final exo scores
            has_exo_score = (
                "exo_score" in df_plot_reg.columns
                and df_plot_reg["exo_score"].notna().any()
            )
            if has_exo_score:
                final_exo_quantiles = (
                    final_scores.groupby(["freq", "sample_window"])["exo_score"]
                    .quantile([0.2, 0.5, 0.8])
                    .unstack()
                    .reset_index()
                )
                final_exo_quantiles = final_exo_quantiles.rename(
                    columns={0.5: "median", 0.2: "q20", 0.8: "q80"}
                )
                print(
                    f"Plotting combined recovery and exo scores for "
                    f"Method={method_val}, Alpha={alpha_val_str}, Reg={reg_val}."
                )
            else:
                print(
                    f"Plotting only recovery score for Method={method_val}, "
                    f"Alpha={alpha_val_str}, Reg={reg_val} "
                    f"(exo_score not found or all NaN)."
                )

            # --- Store quantiles for aggregated plot (within method/alpha) ---
            temp_final_quantiles = final_quantiles.copy()
            temp_final_quantiles["reg"] = reg_val  # Add reg info
            all_final_quantiles.append(temp_final_quantiles)
            if has_exo_score:
                temp_final_exo_quantiles = final_exo_quantiles.copy()
                temp_final_exo_quantiles["reg"] = reg_val  # Add reg info
                all_final_exo_quantiles.append(temp_final_exo_quantiles)
            processed_regs.append(
                reg_val
            )  # Track regs processed for the aggregation step
            # --- End Store quantiles ---

            unique_windows = sorted(
                final_quantiles["sample_window"].unique()
            )  # Based on filtered data for this reg

            if PLOT_PER_WINDOW:
                # Generate one plot per window size
                print(
                    f"Generating separate plots per window size for "
                    f"Method={method_val}, Alpha={alpha_val_str}, Reg={reg_val}."
                )
                for sw in unique_windows:
                    plt.figure(
                        figsize=(10, 6)
                    )  # Slightly smaller figure for single window
                    ax = plt.gca()

                    # Filter data for the current window
                    sw_data = final_quantiles[final_quantiles["sample_window"] == sw]
                    # Plot recovery score (solid line, circle marker)
                    ax.plot(
                        sw_data["freq"],
                        sw_data["median"],
                        color="C0",
                        marker="o",
                        linestyle="-",
                        label="True D Recovery",
                    )
                    ax.fill_between(
                        sw_data["freq"],
                        sw_data["q20"],
                        sw_data["q80"],
                        alpha=0.2,
                        color="C0",
                    )

                    # Plot exo score (dashed line, cross marker) if available
                    if (
                        has_exo_score
                        and sw in final_exo_quantiles["sample_window"].unique()
                    ):
                        sw_exo_data = final_exo_quantiles[
                            final_exo_quantiles["sample_window"] == sw
                        ]
                        ax.plot(
                            sw_exo_data["freq"],
                            sw_exo_data["median"],
                            color="C1",
                            marker="x",
                            linestyle="--",
                            label="Rare event D Recovery",
                        )
                        ax.fill_between(
                            sw_exo_data["freq"],
                            sw_exo_data["q20"],
                            sw_exo_data["q80"],
                            alpha=0.2,
                            color="C1",
                            linestyle="--",
                        )
                    elif has_exo_score:
                        print(
                            f"Warning: No exo_score data for sample_window {sw} "
                            f"with Method={method_val}, "
                            f"Alpha={alpha_val_str}, Reg={reg_val}"
                        )

                    ax.set_xscale("log")  # Set x-axis to log scale
                    plt.xlabel("Frequency of Rare Event (log scale)")
                    plt.ylabel("Final Recovery Score (Median, 20%-80% quantiles)")
                    plt.title(
                        f"Recovery Score vs Freq (Method={method_val}, "
                        f"Alpha={alpha_val_str}, Reg={reg_val}, "
                        f"Window={sw}, Runs={n_runs})"
                    )  # Updated title
                    plt.ylim(0.2, 1.02)  # Keep commented out or adjust as needed
                    plt.legend(title="Score Type")  # Simpler legend
                    plt.tight_layout()  # Adjust layout

                    # Format window size for filename (handle tuples stored as strings)
                    sw_str = str(sw).replace(" ", "").replace(",", "_").strip("()")
                    # Updated filename
                    plot_filename = (
                        exp_dir
                        / (
                            f"final_combined_recovery_vs_freq_meth_{method_val}"
                            f"_alpha_{alpha_val_str}_reg_{reg_val}_ws_{sw_str}.pdf"
                        )
                    )
                    plt.savefig(plot_filename)
                    print(f"Final combined score plot saved to {plot_filename}")
                    plt.close()

            else:
                # Original behavior: Plot all windows on one graph
                print(
                    f"Generating single plot with all window sizes for "
                    f"Method={method_val}, Alpha={alpha_val_str}, Reg={reg_val}."
                )  # Updated print
                plt.figure(figsize=(12, 7))
                ax = plt.gca()
                colors = sns.color_palette("viridis", n_colors=len(unique_windows))
                window_color_map = dict(zip(unique_windows, colors, strict=False))
                lines = []  # For custom legend handles

                for sw in unique_windows:
                    color = window_color_map[sw]
                    # Plot recovery score (solid line, circle marker)
                    sw_data = final_quantiles[final_quantiles["sample_window"] == sw]
                    (line_rec,) = ax.plot(
                        sw_data["freq"],
                        sw_data["median"],
                        color=color,
                        marker="o",
                        linestyle="-",
                        label=f"Window {sw} (True D)",
                    )
                    ax.fill_between(
                        sw_data["freq"],
                        sw_data["q20"],
                        sw_data["q80"],
                        alpha=0.15,
                        color=color,
                    )

                    # Plot exo score (dashed line, cross marker) if available
                    if has_exo_score:
                        # Ensure data exists for this window in exo scores
                        if sw in final_exo_quantiles["sample_window"].unique():
                            sw_exo_data = final_exo_quantiles[
                                final_exo_quantiles["sample_window"] == sw
                            ]
                            (line_exo,) = ax.plot(
                                sw_exo_data["freq"],
                                sw_exo_data["median"],
                                color=color,
                                marker="x",
                                linestyle="--",
                                label=f"Window {sw} (Rare event D)",
                            )
                            ax.fill_between(
                                sw_exo_data["freq"],
                                sw_exo_data["q20"],
                                sw_exo_data["q80"],
                                alpha=0.15,
                                color=color,
                                linestyle="--",
                            )
                        else:
                            print(
                                f"Warning: No exo_score data for sample_window {sw} "
                                f"with Method={method_val}, Alpha={alpha_val_str}, "
                                f"Reg={reg_val}"
                            )

                # Create custom legend handles
                # Handles for score type (linestyle/marker)
                score_handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color="gray",
                        linestyle="-",
                        marker="o",
                        label="True D Recovery",
                    )
                ]
                if has_exo_score:
                    score_handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color="gray",
                            linestyle="--",
                            marker="x",
                            label="Rare event D Recovery",
                        )
                    )

                # Handles for sample window (color)
                window_handles = [
                    plt.Line2D(
                        [0], [0], color=window_color_map[sw], lw=4, label=f"Window {sw}"
                    )
                    for sw in unique_windows
                ]

                ax.set_xscale("log")  # Set x-axis to log scale
                plt.xlabel("Frequency of Rare Event (log scale)")
                plt.ylabel("Final Recovery Score (Median, 20%-80% quantiles)")
                plt.title(
                    f"Dictionary Recovery Score vs Frequency "
                    f"(Method={method_val}, Alpha={alpha_val_str}, "
                    f"Reg={reg_val}, Runs={n_runs})"
                )
                plt.ylim(0.2, 1.02)  # Adjust y-axis limits (start from 0 for exo)

                # Create combined legend
                # Place score type legend
                leg1 = plt.legend(
                    handles=score_handles,
                    title="Score Type",
                    loc="lower right",
                    bbox_to_anchor=(1.27, 0),
                )
                # Place window legend
                leg2 = plt.legend(
                    handles=window_handles,
                    title="Sample Window",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )
                # Add the first legend back because the second one replaces it
                ax.add_artist(leg1)

                plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legends
                # Updated filename
                plot_filename = (
                    exp_dir
                    / (
                        f"final_combined_recovery_vs_freq_meth_{method_val}"
                        f"_alpha_{alpha_val_str}_reg_{reg_val}.pdf"
                    )
                )
                plt.savefig(plot_filename)
                print(f"Final combined score plot saved to {plot_filename}")
                plt.close()
        else:
            print(
                "Skipping combined recovery_score plot for "
                f"Method={method_val}, Alpha={alpha_val_str}, Reg={reg_val} "
                "as 'recovery_score' column not found."
            )

    # --- Plot recovery score curves over epochs
    # (still within method/alpha, but potentially across regs if desired) ---
    # Current implementation plots curves for all regs within the
    # method/alpha subset on the same FacetGrid.
    # If separation by reg is needed here too, FacetGrid could use 'reg'
    # for rows or another dimension.
    # Using df_plot which contains all regs for the current method/alpha.
    if "recovery_score" in df_plot.columns:
        # Median and quantiles per freq, epoch, sample_window (and potentially reg)
        grouping_cols = ["freq", "epoch", "sample_window"]
        curves = (
            df_plot.groupby(grouping_cols)["recovery_score"]
            .quantile([0.2, 0.5, 0.8])
            .unstack()
            .reset_index()
        )
        curves = curves.rename(columns={0.5: "median", 0.2: "q20", 0.8: "q80"})

        if curves.empty:
            print(
                "Skipping recovery curve plot for "
                f"Method={method_val}, Alpha={alpha_val_str} as no data remains "
                "after filtering/grouping."
            )
            continue

        # Use seaborn FacetGrid for clarity across sample_windows
        num_unique_windows_plot = curves["sample_window"].nunique()
        # Decide hue: currently 'freq'. Could be changed to 'reg' if desired.
        g = sns.FacetGrid(
            curves,
            col="sample_window",
            col_wrap=min(3, num_unique_windows_plot),
            hue="freq",
            palette="viridis",
            sharey=True,
            height=4,
        )
        g.map_dataframe(sns.lineplot, x="epoch", y="median")

        # Add shaded areas (iterating through axes)
        for ax in g.axes_dict.values():
            title_parts = ax.get_title().split(" = ")
            if len(title_parts) == 2:
                sw_val_str = title_parts[1]
                sw_dtype = df_plot["sample_window"].dtype
                try:
                    if pd.api.types.is_numeric_dtype(sw_dtype):
                        sw_val = pd.to_numeric(sw_val_str)
                    else:
                        sw_val = sw_val_str
                except ValueError:
                    sw_val = sw_val_str
            else:
                print(
                    "Warning: Could not parse sample_window from title '"
                    f"{ax.get_title()}'. Skipping quantile shading."
                )
                continue

            curves_sw = curves[curves["sample_window"] == sw_val]
            # Determine the unique elements for the hue dimension (e.g., 'freq')
            hue_col_name = "freq"  # Explicitly use the hue column name
            hue_elements = sorted(curves_sw[hue_col_name].unique())
            hue_order = g.hue_names  # Get the order used by FacetGrid
            palette = sns.color_palette("viridis", n_colors=len(hue_order))
            color_map = dict(zip(hue_order, palette, strict=False))

            for hue_val in hue_elements:
                if hue_val in color_map:  # Check if hue value is plotted
                    # Filter based on the hue dimension name
                    hue_curve = curves_sw[curves_sw[hue_col_name] == hue_val]
                    ax.fill_between(
                        hue_curve["epoch"],
                        hue_curve["q20"],
                        hue_curve["q80"],
                        alpha=0.2,
                        color=color_map[hue_val],
                    )

        g.set_axis_labels("Epoch", "Median Recovery Score (20%-80% quantiles)")
        g.set_titles(col_template="Sample Window = {col_name}")
        g.add_legend(
            title=hue_col_name.capitalize()
        )  # Use hue dimension name for legend title
        g.fig.suptitle(
            "Recovery Score during Training "
            f"(Method={method_val}, Alpha={alpha_val_str}, Runs={n_runs})",
            y=1.03,
        )  # Updated title
        g.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout

        # Updated filename
        curve_plot_filename = (
            exp_dir
            / f"recovery_curves_freq_meth_{method_val}_alpha_{alpha_val_str}.pdf"
        )
        plt.savefig(curve_plot_filename)
        print(f"Recovery curve plot saved to {curve_plot_filename}")
        plt.close()
    else:
        print(
            f"Skipping recovery curve plot for Method={method_val}, "
            f"Alpha={alpha_val_str} as 'recovery_score' column not found."
        )

    # --- AGGREGATED PLOTTING SECTION:
    # Final Scores vs Freq, colored by Reg (within Method/Alpha) ---
    # This section now runs inside the main loop for each (method, alpha) pair.

    print(
        "\nGenerating aggregated plots for "
        f"Method={method_val}, Alpha={alpha_val_str}: "
        "Final Scores vs Freq (colored by Reg)"
    )

    if not all_final_quantiles:  # Check if any data was collected for this method/alpha
        print(
            f"Skipping aggregated plots for Method={method_val}, "
            f"Alpha={alpha_val_str} as no data was collected "
            "(check filters and input data)."
        )
        continue  # Skip to the next method/alpha pair

    # Concatenate collected quantiles for the current method/alpha
    agg_final_quantiles = pd.concat(all_final_quantiles, ignore_index=True)
    agg_has_exo_score = bool(
        all_final_exo_quantiles
    )  # Check if any exo data was collected
    if agg_has_exo_score:
        agg_final_exo_quantiles = pd.concat(all_final_exo_quantiles, ignore_index=True)

    # Unique windows and regs found *within* this method/alpha subset's processed data
    agg_unique_windows = sorted(agg_final_quantiles["sample_window"].unique())
    agg_unique_regs = sorted(
        agg_final_quantiles["reg"].unique()
    )  # Use regs present in the collected data

    # Create one plot per window size
    for sw in agg_unique_windows:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        # Filter aggregated data for the current window
        win_quantiles = agg_final_quantiles[agg_final_quantiles["sample_window"] == sw]
        if agg_has_exo_score:
            # Ensure the exo df is not empty before filtering
            if not agg_final_exo_quantiles.empty:
                win_exo_quantiles = agg_final_exo_quantiles[
                    agg_final_exo_quantiles["sample_window"] == sw
                ]
            else:
                win_exo_quantiles = pd.DataFrame()  # Empty df if no exo data at all

        # Define colors for regularization values using tab10
        reg_colors = sns.color_palette("tab10", n_colors=len(agg_unique_regs))
        reg_color_map = dict(zip(agg_unique_regs, reg_colors, strict=False))

        # Plot data for each regularization value
        for reg in agg_unique_regs:
            color = reg_color_map[reg]

            # Plot recovery_score (dashed line)
            reg_win_data = win_quantiles[win_quantiles["reg"] == reg]
            if not reg_win_data.empty:
                ax.plot(
                    reg_win_data["freq"],
                    reg_win_data["median"],
                    color=color,
                    marker="o",
                    markersize=4,
                    linestyle="--",
                    label=f"Reg {reg} (True D)",
                )
                ax.fill_between(
                    reg_win_data["freq"],
                    reg_win_data["q20"],
                    reg_win_data["q80"],
                    alpha=0.1,
                    color=color,
                    linestyle="--",
                )

            # Plot exo_score (solid line)
            # Check if win_exo_quantiles is not empty before proceeding
            if agg_has_exo_score and not win_exo_quantiles.empty:
                reg_win_exo_data = win_exo_quantiles[win_exo_quantiles["reg"] == reg]
                if not reg_win_exo_data.empty:
                    ax.plot(
                        reg_win_exo_data["freq"],
                        reg_win_exo_data["median"],
                        color=color,
                        marker="x",
                        markersize=5,
                        linestyle="-",
                        label=f"Reg {reg} (Rare event D)",
                    )
                    ax.fill_between(
                        reg_win_exo_data["freq"],
                        reg_win_exo_data["q20"],
                        reg_win_exo_data["q80"],
                        alpha=0.1,
                        color=color,
                        linestyle="-",
                    )

        ax.set_xscale("log")
        plt.xlabel("Frequency of Rare Event (log scale)")
        plt.ylabel("Final Recovery Score (Median, 20%-80% quantiles)")
        # Format window size for title (handle potential tuples stored as strings)
        sw_title_str = str(sw).strip("()")
        # Updated title
        plt.title(
            "Final Score vs Frequency "
            f"(Method={method_val}, Alpha={alpha_val_str}, "
            f"Window={sw_title_str}, Runs={n_runs})"
        )
        plt.ylim(0.2, 1.02)

        # Create custom legends
        # Legend for Score Type (linestyle)
        score_lines = [
            mlines.Line2D(
                [],
                [],
                color="gray",
                linestyle="--",
                marker="o",
                markersize=4,
                label="True D Recovery",
            )
        ]
        if agg_has_exo_score:
            score_lines.append(
                mlines.Line2D(
                    [],
                    [],
                    color="gray",
                    linestyle="-",
                    marker="x",
                    markersize=5,
                    label="Rare event D Recovery",
                )
            )

        # Legend for Regularization (color)
        reg_lines = [
            mlines.Line2D([], [], color=reg_color_map[reg], lw=2, label=f"Reg {reg}")
            for reg in agg_unique_regs
        ]

        # Place legends stacked vertically outside the plot
        leg2 = plt.legend(
            handles=reg_lines,
            title="Regularization",
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
        )
        leg1 = plt.legend(
            handles=score_lines,
            title="Score Type",
            loc="upper left",
            bbox_to_anchor=(1.05, 0.8),
        )  # Adjust y-anchor if needed
        ax.add_artist(leg2)  # Add the regularization legend back

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legends

        # Format window size for filename
        sw_file_str = str(sw).replace(" ", "").replace(",", "_").strip("()")
        # Updated filename
        agg_plot_filename = (
            exp_dir
            / (
                f"agg_final_score_vs_freq_meth_{method_val}"
                f"_alpha_{alpha_val_str}_ws_{sw_file_str}.pdf"
            )
        )
        plt.savefig(agg_plot_filename)
        print(f"Aggregated final score plot saved to {agg_plot_filename}")
        plt.close()

print("\n--- Plotting finished ---")
# %%
