# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils_apnea import load_ecg, plot_1d_trials

# %matplotlib inline

sns.reset_defaults()
plt.rcdefaults()

df = pd.read_csv("apnea-ecg/df_std.tsv", sep="\t")
df_none = df[df["cutoff"] == "None"]

# for subject_id in df_none["subject_id"].unique():
#     for alpha in [1, 2.5, 5]:
#         df_cutoff = df_none[df_none["subject_id"] == subject_id].copy()
#         q1, q99 = np.percentile(df_cutoff["std"].values, [alpha, 100 - alpha])
#         df_cutoff.drop(
#             df_cutoff[(df_cutoff["std"] > q99) | (df_cutoff["std"] < q1)].index,
#             inplace=True,
#         )
#         df_cutoff["cutoff"] = f"alpha = {alpha}%"
#         df = pd.concat([df, df_cutoff], ignore_index=True)

# df.to_csv("apnea-ecg/df_std.tsv", sep="\t", index=False)

# 1 / 0
# %%

subject_id = "a05"
X, all_labels = load_ecg(
    subject_id,
    split=True,
    T=60,
    apply_window=False,
    verbose=False,
)

df_subject = df_none[df_none["subject_id"] == subject_id]
for alpha in [1]:
    q1, q99 = np.percentile(df_subject["std"].values, [alpha, 100 - alpha])
    df_subject_alpha = df_subject[(df_subject["std"] > q99) | (df_subject["std"] < q1)]
    trial_id = df_subject_alpha["trial_id"].values
    std_values = df_subject_alpha["std"].values
    ylabels = [f"{id} ({std:.3})" for id, std in zip(trial_id, std_values)]
    fig = plot_1d_trials(X[trial_id], ylabels=ylabels)
    fig.axes[0].set_title(f"Subject {subject_id}, extreme std trials, alpha = {alpha}%")
    plt.xlim(0, 6000)
    plt.show()

# %% Plot number of trials remaining after cutoff, for each subject, by group
for group_id in ["a", "b", "c", "x"]:
    df_group = df[df["subject_id"].str.startswith(group_id)]
    df_group = df_group.groupby(["cutoff", "subject_id"]).count()
    df_group = df_group.rename(columns={"trial_id": "n_trials"})
    df_group.drop(["labels", "std"], axis="columns", inplace=True)
    df_group.reset_index()
    sns.lineplot(df_group, x="subject_id", y="n_trials", hue="cutoff")
    plt.xticks(rotation=90, ha="right")
    plt.xlabel("Subject ID")
    plt.ylabel("Number of trials")
    plt.title(f"Group {group_id}")
    plt.show()
    plt.clf()

# %% Plot the distribution of trials' std with(out) cutoff, for each subject, by group

for group_id in ["a", "b", "c", "x"]:
    df_group = df[df["subject_id"].str.startswith(group_id)]
    n_subjects = len(df_group["subject_id"].unique())

    fig, ax = plt.subplots(figsize=(7, 0.8 * n_subjects))
    alpha = 1
    sns.boxplot(
        data=df_group,
        x="std",
        y="subject_id",
        hue="cutoff",
        whis=[alpha, 100 - alpha],
        width=0.8,
    )
    plt.xlim(-0.1, None)
    plt.xscale("symlog")
    plt.xlabel("Trials standard deviation")
    plt.ylabel("Subject ID")
    plt.title(f"Group {group_id}")
    plt.show()
    plt.clf()
