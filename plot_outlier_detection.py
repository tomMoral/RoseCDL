"""This file reads the results from outlier_detection.py and generates the plots for the
outliers detection task using the WinCDL algorithm.
"""

# %%
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_method_name(outliers_kwargs):
    if outliers_kwargs is None:
        return "no detection"

    method = outliers_kwargs["method"]
    if "unilateral" in method:
        method = method.split("_")[0]
    if "alpha" in outliers_kwargs:
        alpha = outliers_kwargs["alpha"]
        method += f" (alpha={alpha})"

    return method


EXP_DIR = Path("outliers_detection")

add_reg = True
per_patch = True
q_lmbd = "method"
REG = 0.8

D_INIT = "random"  # "random" or "chunk"

n_runs = 20

exp_name = f"add_reg={add_reg}_per_patch={per_patch}_q_lmbd={q_lmbd}_reg={REG}_D_init={D_INIT}_normalized_error_metric_no_opening_window"
exp_dir = EXP_DIR / exp_name
exp_dir.mkdir(exist_ok=True, parents=True)

df_scores = pd.read_csv(exp_dir / "df_results.csv")
df_scores = df_scores.rename(
    columns=dict(
        iteration="Iteration",
        method="Method",
        accuracy="Accuracy",
        precision="Precision",
        recall="Recall",
        f1="F1 score",
        dice="Dice score",
        jaccard="Jaccard score",
    )
)

# %%

alpha_true = 0.1
list_methods = list_methods = [
    None,
    {"method": "quantile_unilateral", "alpha": alpha_true * 2},
    {"method": "quantile_unilateral", "alpha": alpha_true},
    {"method": "quantile_unilateral", "alpha": alpha_true / 2},
    {"method": "iqr_unilateral", "alpha": 1.5},
    {"method": "zscore", "alpha": 1},
    {"method": "zscore", "alpha": 2},
    # {"method": "zscore", "alpha": 3},
    # {"method": "mad", "alpha": 1},
    {"method": "mad", "alpha": 3.5},
]

ADD_TITLES = False

list_names = [get_method_name(kwargs) for kwargs in list_methods]
# Only keep methods that are in the list
df_scores = df_scores[df_scores["Method"].isin(list_names)]
methods = df_scores["Method"].unique()
print(f"Methods kept: {methods}")

# Replace 'alpha' with LaTeX-style symbol
df_scores["Method"] = df_scores["Method"].str.replace("quantile", "Quant.", regex=False)
df_scores["Method"] = df_scores["Method"].str.replace("alpha", r"$\alpha$", regex=False)
df_scores["Method"] = df_scores["Method"].str.replace("zscore", "z-score", regex=False)
df_scores["Method"] = df_scores["Method"].str.replace("mad", "MAD", regex=False)
df_scores["Method"] = df_scores["Method"].str.replace("iqr", "IQR", regex=False)
df_scores["Method"] = df_scores["Method"].str.replace(
    "no detection", "No detection", regex=False
)

df_final = df_scores[df_scores["Iteration"] == df_scores["Iteration"].max()]

# Compute how many repetitions has been computed
n_runs = df_scores["seed"].nunique()
print(f"n_runs = {n_runs}")

# Define method order
methods = df_scores["Method"].unique()
quantile_methods = sorted([m for m in methods if "Quant." in m])
iqr_methods = sorted([m for m in methods if "IQR" in m])
zscore_methods = sorted([m for m in methods if "z-score" in m])
mad_methods = sorted([m for m in methods if "MAD" in m])
method_order = quantile_methods + iqr_methods + zscore_methods + mad_methods

other_methods = [m for m in methods if m not in method_order]

# %%

# Set seaborn style for paper plots
sns.set(style="whitegrid", context="paper", font_scale=1.1)
sns.set_style("whitegrid")

fig_width = 8.5 / 2
figsize = (fig_width, fig_width)

# Plot final scores in boxplot
print("Plotting score boxplot... ", end="", flush=True)
sns.set(style="whitegrid", context="paper", font_scale=1.1)
sns.set_style("whitegrid")
# specify figure size
plt.figure(figsize=figsize)
sns.boxplot(
    data=df_final,
    y="Method",
    x="score",
    showmeans=True,
    orient="h",
    meanprops={
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "5",
    },
    order=method_order + other_methods,
)
plt.ylabel("Detection method")
plt.xlabel("Recovery score")
if ADD_TITLES:
    plt.title(f"Recovery score for different detection methods ({n_runs} repetitions)")
plt.tight_layout()
plt.savefig(exp_dir / "boxplot_score_detection.pdf")
plt.show()
plt.close()
print("Done")

# Plot final percentage
print("Plotting percentage boxplot... ", end="", flush=True)
sns.set(style="whitegrid", context="paper", font_scale=1.1)
sns.set_style("whitegrid")
plt.figure(figsize=figsize)
sns.boxplot(
    data=df_final[df_final["Method"] != "No detection"],
    y="Method",
    x="effective_percentage",
    showmeans=True,
    orient="h",
    meanprops={
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "5",
    },
    order=method_order,
)
# Add a vertical line at 10%
plt.axvline(x=0.1, color="red", linestyle="--")
plt.ylabel("Detection method")
plt.xlabel("Outlier percentage")
if ADD_TITLES:
    plt.title(
        f"Outlier percentage for different detection methods ({n_runs} repetitions)"
    )
plt.tight_layout()
plt.savefig(exp_dir / "boxplot_percentage_detection.pdf")
plt.show()
plt.close()
print("Done")

# Plot final metrics in boxplot
print("Plotting metric boxplot... ", end="", flush=True)

# Reshape the DataFrame
df_final_metric = df_final[df_final["Method"] != "No detection"]
df_final_metric = df_final_metric.melt(
    id_vars=["Method"],
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 score",
        "Dice score",
        "Jaccard score",
    ],
    var_name="Metric",
    value_name="Value",
)

sns.set(style="whitegrid", context="paper", font_scale=1.1)
sns.set_style("whitegrid")
plt.figure(figsize=figsize)
sns.boxplot(
    data=df_final_metric,
    y="Method",
    x="Value",
    hue="Metric",
    showmeans=True,
    orient="h",
    meanprops={
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "5",
    },
    order=method_order,
)
plt.ylabel("Detection method")
plt.xlabel("Metric value")
if ADD_TITLES:
    plt.title(f"Metric value for different detection methods ({n_runs} repetitions)")
plt.tight_layout()
plt.savefig(exp_dir / "boxplot_metric_detection.pdf")
plt.show()
plt.close()
print("Done")

# Plot metrics in different subplots
print("Plotting metric separeted boxplots... ", end="", flush=True)
# Metrics
metrics = df_final_metric["Metric"].unique()

# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=len(metrics), figsize=(8, 12), sharex=True)

# Create a boxplot for each metric
for ax, metric in zip(axes, metrics):
    sns.set(style="whitegrid", context="paper", font_scale=1.1)
    sns.set_style("whitegrid")
    sns.boxplot(
        data=df_final_metric[df_final_metric["Metric"] == metric],
        y="Method",
        x="Value",
        ax=ax,
        showmeans=True,
        orient="h",
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "5",
        },
        order=method_order,
    )
    ax.set_title(metric)

# Adjust layout
plt.tight_layout()
plt.savefig(exp_dir / "boxplot_metric_separeted_detection.pdf")
plt.show()
plt.close()
print("Done")

# %%
###############################################################################
# Plotting the evolution of the recovery score and metrics as a function of iterations
###############################################################################

errorbar = ("ci", 80)

df_scores["Method"] = pd.Categorical(
    df_scores["Method"], categories=method_order + other_methods, ordered=True
)


print("Plotting recovery scores... ", end="", flush=True)
sns.set(style="white", context="paper", font_scale=1.1)

plt.figure(figsize=figsize)
sns.lineplot(
    data=df_scores, x="Iteration", y="score", hue="Method", errorbar=errorbar, alpha=0.6
)
plt.xlabel("Iteration")
plt.ylabel("Recovery score")
if ADD_TITLES:
    plt.title(f"Recovery score as a function of iterations ({n_runs} repetitions)")
plt.xlim(0, None)
plt.tight_layout()
plt.savefig(exp_dir / "score_detection.pdf")
plt.show()
plt.close()
print("Done")

print("Plotting percentage evolution... ", end="", flush=True)
sns.set(style="whitegrid", context="paper", font_scale=1.1)
sns.set_style("whitegrid")
plt.figure(figsize=figsize)
sns.lineplot(
    data=df_scores[df_scores["Method"] != "No detection"],
    x="Iteration",
    y="effective_percentage",
    hue="Method",
    errorbar=errorbar,
    alpha=0.6,
)
plt.hlines(
    y=0.1,
    xmin=0,
    xmax=df_scores["Iteration"].max(),
    color="red",
    linestyle="--",
    label="True contamination percentag (10%)",
)
plt.xlabel("Iteration")
plt.ylabel("Outlier percentage ")
if ADD_TITLES:
    plt.title(f"Outlier percentage as a function of iterations ({n_runs} repetitions)")
plt.xlim(0, None)
plt.tight_layout()
plt.savefig(exp_dir / "percentage_detection.pdf")
plt.show()
plt.close()
print("Done")

# Plot metrics as a function of iterations
print("Plotting metrics... ", end="", flush=True)
sns.set(style="whitegrid", context="paper", font_scale=1.1)
sns.set_style("whitegrid")

# Dispatch all metrics over 2 columns
n_rows, n_cols = math.ceil(len(metrics) / 2), 2
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.flatten()  # Flatten the axes array for easy indexing
# Plot each metric in a subplot
for ax, this_metric in zip(axes, metrics):
    sns.lineplot(
        data=df_scores[df_scores["Method"] != "No detection"],
        x="Iteration",
        y=this_metric,
        hue="Method",
        errorbar="ci",
        ax=ax,
    )
    ax.set_title(this_metric)
    if ax is not axes[0]:
        # no legend
        ax.get_legend().remove()
if ADD_TITLES:
    plt.suptitle(f"Metrics Evolution Over Iterations by Method ({n_runs} repetitions)")
plt.xlim(0, None)
plt.tight_layout()
plt.savefig(exp_dir / "metrics_detection.pdf")
plt.show()
plt.close()
print("Done, saved in", exp_dir / "metrics_evolution.pdf")
# %%

# Plot each metric in a plot
df_temp = df_scores[df_scores["Method"] != "No detection"]
unique_methods = df_temp["Method"].unique()
unique_methods = [
    r"Quant. ($\alpha$=0.05)",
    r"Quant. ($\alpha$=0.1)",
    r"Quant. ($\alpha$=0.2)",
    r"IQR ($\alpha$=1.5)",
    r"z-score ($\alpha$=1)",
    r"z-score ($\alpha$=2)",
    r"MAD ($\alpha$=3.5)",
]
# Restart seaborn


print("Plotting metrics... ", end="", flush=True)
sns.set(style="white", context="paper", font_scale=1.1)

estimator = "median"
errorbar = "se"  # ("pi", 50) ("ci", 80)  "se"


for this_metric in metrics:
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=df_temp,
        x="Iteration",
        y=this_metric,
        hue="Method",
        estimator=estimator,
        errorbar=errorbar,
        hue_order=unique_methods,
    )

    # plt.title(this_metric)
    # Remove legend
    # plt.legend([],[], frameon=False)
    plt.xlim(0, None)
    plt.tight_layout()
    plt.savefig(exp_dir / f"{this_metric}_detection.pdf")
    plt.show()
    plt.close()
