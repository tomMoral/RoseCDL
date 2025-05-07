# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# %%
# --- Configuration ---
# TODO: Update with the correct path to your results CSV
CSV_PATH = "../results/rare_event_detection/results.csv"  # Placeholder path
# TODO: Update with the desired output directory
OUTPUT_DIR = Path("../results/rare_event_detection/")
# --- End Configuration ---

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the main DataFrame
try:
    main_df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded data from {CSV_PATH}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_PATH}")
except Exception as e:
    print(f"Error loading CSV: {e}")

print("DataFrame head:")
print(main_df.head())
print("\nDataFrame columns:")
print(main_df.columns)
print("\nDataFrame info:")
main_df.info()


# %%
# --- Data Validation and Preparation ---

required_columns = [
    "epoch",
    "precision",
    "recall",
    "f1",
    "method",
    "alpha",
    "opening_window",
    "seed",
]
missing_cols = [col for col in required_columns if col not in main_df.columns]

if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}")
    # Handle error appropriately, e.g., exit or try defaults
    # For now, we'll exit
    sys.exit()
else:
    print("All required columns found.")

# Calculate n_runs (number of seeds)
if "seed" in main_df.columns:
    n_runs = main_df["seed"].nunique()
    print(f"Found {n_runs} unique seeds (runs).")
else:
    # This case should be caught by the check above, but as a fallback:
    n_runs = 1
    main_df["seed"] = 0  # Add a dummy seed if needed
    print("Warning: 'seed' column not found. Assuming n_runs=1.")

# Convert alpha to string for consistent grouping/filenames if it's numeric
if pd.api.types.is_numeric_dtype(main_df["alpha"]):
    main_df["alpha_str"] = main_df["alpha"].astype(str)
    alpha_col_name = "alpha_str"
else:
    main_df["alpha_str"] = main_df["alpha"].astype(str)  # Ensure it's string anyway
    alpha_col_name = "alpha_str"

# Convert opening_window to string if it's not already
if not pd.api.types.is_string_dtype(main_df["opening_window"]):
    main_df["opening_window_str"] = main_df["opening_window"].astype(str)
    window_col_name = "opening_window_str"
else:
    main_df["opening_window_str"] = main_df["opening_window"]  # Ensure column exists
    window_col_name = "opening_window_str"


# --- Get unique method, alpha, and opening_window combinations ---
grouping_cols = ["method", alpha_col_name, window_col_name]
try:
    unique_combinations = main_df[grouping_cols].drop_duplicates().to_numpy().tolist()
    print(
        f"Found {len(unique_combinations)} unique "
        f"(method, alpha, opening_window) combinations."
    )
    print(f"Combinations: {unique_combinations}")
except KeyError as e:
    print(f"Error finding unique combinations. Missing column: {e}")

# %%
# --- Plotting Section ---

print("\n--- Starting Plot Generation ---")

for method_val, alpha_val_str, window_val_str in unique_combinations:
    print(
        f"\nProcessing: Method={method_val}, "
        f"Alpha={alpha_val_str}, Window={window_val_str}"
    )

    # Filter DataFrame for the current combination
    df_subset = main_df[
        (main_df["method"] == method_val)
        & (main_df[alpha_col_name] == alpha_val_str)
        & (main_df[window_col_name] == window_val_str)
    ].copy()

    if df_subset.empty:
        print("  Skipping: No data for this combination.")
        continue

    print(f"  Data points for this combination: {len(df_subset)}")

    # Calculate quantiles over epochs across different seeds
    metrics_to_plot = ["precision", "recall", "f1"]
    try:
        # Ensure metrics are numeric
        for metric in metrics_to_plot:
            df_subset[metric] = pd.to_numeric(df_subset[metric], errors="coerce")
        df_subset = df_subset.dropna(
            subset=metrics_to_plot
        )  # Drop rows where metrics are NaN after conversion

        if df_subset.empty:
            print(
                "  Skipping: No valid numeric metric data after coercion/dropping NaNs."
            )
            continue

        curves = df_subset.groupby(["epoch"])[metrics_to_plot].agg(
            [
                ("q20", lambda x: x.quantile(0.2)),
                ("median", "median"),
                ("q80", lambda x: x.quantile(0.8)),
            ]
        )
        # Flatten MultiIndex columns
        curves.columns = ["_".join(col).strip() for col in curves.columns.to_numpy()]
        curves = curves.reset_index()  # Make 'epoch' a column again

    except KeyError as e:
        print(f"  Skipping: Error calculating quantiles. Missing metric column? {e}")
        continue
    except Exception as e:
        print(
            f"  Skipping: An unexpected error occurred during quantile calculation: {e}"
        )
        continue

    if curves.empty:
        print("  Skipping: No data after grouping by epoch.")
        continue

    # --- Create Plot ---
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), sharey=True
    )  # 1 row, 3 columns for metrics
    fig.suptitle(
        "Performance Metrics vs Epoch "
        f"(Method={method_val}, Alpha={alpha_val_str}, "
        f"Window={window_val_str}, Runs={n_runs})",
        fontsize=14,
        y=1.02,
    )

    plot_configs = [
        {"metric": "precision", "ax_idx": 0, "title": "Precision"},
        {"metric": "recall", "ax_idx": 1, "title": "Recall"},
        {"metric": "f1", "ax_idx": 2, "title": "F1 Score"},
    ]

    for config in plot_configs:
        metric = config["metric"]
        ax = axes[config["ax_idx"]]
        median_col = f"{metric}_median"
        q20_col = f"{metric}_q20"
        q80_col = f"{metric}_q80"

        if (
            median_col not in curves.columns
            or q20_col not in curves.columns
            or q80_col not in curves.columns
        ):
            print(
                f"  Warning: Quantile columns missing for metric '{metric}'."
                f" Skipping plot for this metric."
            )
            ax.set_title(f"{config['title']} (Data Missing)")
            ax.text(
                0.5,
                0.5,
                "Data Missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            continue

        # Plot median line
        ax.plot(
            curves["epoch"],
            curves[median_col],
            label=f"Median {config['title']}",
            color=f"C{config['ax_idx']}",
        )
        # Plot shaded quantile area
        ax.fill_between(
            curves["epoch"],
            curves[q20_col],
            curves[q80_col],
            alpha=0.2,
            color=f"C{config['ax_idx']}",
            label="20%-80% Quantiles",
        )

        ax.set_xlabel("Epoch")
        if config["ax_idx"] == 0:  # Only set y-label for the first plot
            ax.set_ylabel("Score")
        ax.set_title(config["title"])
        ax.set_ylim(0.0, 1.02)  # Set Y-axis limits
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to prevent title overlap

    # --- Save Plot ---
    # Sanitize window value for filename
    window_file_str = str(window_val_str).replace(" ", "").replace(",", "_").strip("()")
    # Sanitize alpha value for filename
    alpha_file_str = str(alpha_val_str).replace(
        ".", "p"
    )  # Replace decimal point if present
    plot_filename = OUTPUT_DIR / (
        f"metrics_vs_epoch_meth_{method_val}_"
        f"alpha_{alpha_file_str}_"
        f"win_{window_file_str}.pdf"
    )
    try:
        plt.savefig(plot_filename, bbox_inches="tight")
        print(f"  Plot saved to {plot_filename}")
    except Exception as e:
        print(f"  Error saving plot {plot_filename}: {e}")
    plt.close(fig)  # Close the figure to free memory

print("\n--- Plotting finished ---")
