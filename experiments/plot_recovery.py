import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot recovery")
    parser.add_argument("results_path", type=str, help="Input file")
    parser.add_argument(
        "--output", "-o", type=str, help="Output directory", default="."
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        help="Methods to plot (format: method;alpha;timing)",
        default=[],
    )

    args = parser.parse_args()

    exp_dir = Path(args.output)
    df_results = pd.read_csv(args.results_path)

    # Apply filters based on method specifications
    if args.methods:
        # Create a filter mask for each method specification
        method_filters = []
        for method_spec in args.methods:
            if method_spec.lower() == "no detection":
                # Special case for no detection
                method_filter = df_results["name"].str.contains(
                    "no detection", case=False
                )
                method_filters.append(method_filter)
            else:
                # Parse method specification (method;alpha;timing)
                parts = method_spec.split(";")
                if len(parts) != 3:
                    print(
                        f"Invalid method specification: {method_spec}. Format should be method;alpha;timing",
                        file=sys.stderr,
                    )

            method, alpha, timing = parts
            # Create combined filter for this method specification
            method_filter = (
                (df_results["method"] == method)
                & (df_results["alpha"] == float(alpha))
                & df_results["name"].str.contains(
                    f"({timing})", case=False, regex=False
                )
            )
            method_filters.append(method_filter)

        # Combine all method filters with OR
        if method_filters:
            df_results = df_results[pd.concat(method_filters, axis=1).any(axis=1)]

    # Check if we have any data after filtering
    if df_results.empty:
        print("No data matches the specified criteria!", file=sys.stderr)
        sys.exit(1)

    # Plot recovery score
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    curves = (
        df_results.groupby(["name", "epoch"])["recovery_score"]
        .quantile([0.2, 0.5, 0.8])
        .unstack()
    )
    for i, name in enumerate(df_results.name.unique()):
        ax.fill_between(
            curves.loc[name].index,
            curves.loc[name, 0.2],
            curves.loc[name, 0.8],
            alpha=0.3,
            color=f"C{i}",
            label=None,
        )
        curves.loc[name, 0.5].plot(label=name, c=f"C{i}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recovery score")
    ax.legend()
    fig.savefig(exp_dir / "recovery_score.pdf")
