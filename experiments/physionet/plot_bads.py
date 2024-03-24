# %%
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from utils_apnea import load_ecg, plot_1d_trials

with open("apnea-ecg/bad_trials.json", "r") as f:
    bads = json.load(f)

participants = pd.read_csv("apnea-ecg/participants.tsv", sep="\t")
df_std = pd.DataFrame()
df_std = pd.read_csv("apnea-ecg/df_std.tsv", sep="\t")

# %%
import mne

subject_id = "b05"
X, all_labels = load_ecg(
    subject_id,
    split=True,
    T=60,
    apply_window=False,
    verbose=False,
)
fs = 100
dict_trials = {
    "a02": range(5, 10),
    "b02": [36, 38, 77, 341, 513],
    "b04": range(5),
    "b05": [142, 145, 148, 172, 175],
}
trials_id = dict_trials.get(subject_id, range(10))
l_freq, h_freq = 0.5, 20
x = X[trials_id]
x_ = mne.filter.filter_data(x, sfreq=fs, l_freq=l_freq, h_freq=h_freq)
fig = plot_1d_trials(
    x, x_, ylabels=trials_id, labels=["original", f"filtered {l_freq}-{h_freq} Hz"]
)
fig.axes[0].set_title(f"Subject {subject_id}, effect of {l_freq}-{h_freq} Hz filtering")
plt.xlim(0, 6000)
plt.show()

# %%

subject_id = "b01"
for l_freq, h_freq in [(None, None), (5, 100)]:
    print(f"filter between {l_freq} and {h_freq}")

    # for subject_id in participants["Record"].values:
    print(f"Subject {subject_id}")
    # if subject_id in df_std["subject_id"].values:
    #     continue
    this_bads = bads.get(subject_id, [])
    X, all_labels = load_ecg(
        subject_id,
        split=True,
        T=60,
        apply_window=False,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=False,
    )
    n_trials = X.shape[0]
    this_std = dict(
        subject_id=np.repeat(subject_id, n_trials),
        trial_id=np.arange(n_trials),
        labels=all_labels,
        std=X.squeeze().std(axis=1),
    )

    sns.histplot(data=pd.DataFrame(this_std), x="std")
    plt.title(f"Subject {subject_id}'s trials standard deviation")
    plt.show()
    # df_std = pd.concat([df_std, pd.DataFrame(this_std)], ignore_index=True)
    # df_std.to_csv("apnea-ecg/df_std.tsv", sep="\t", index=False)

    if len(this_bads) > 0:
        colors = np.repeat(None, n_trials)
        colors[this_bads] = "red"
    else:
        colors = np.repeat(None, n_trials)
    for i in tqdm(range(n_trials // 200 + 1)):
        first_trial, last_trial = i * 200, min((i + 1) * 200, n_trials)
        this_labels = [
            f"Trial {i} ({all_labels[i]})" for i in np.arange(first_trial, last_trial)
        ]
        fig = plot_1d_trials(
            X.squeeze()[first_trial:last_trial],
            labels=this_labels,
            colors=colors[first_trial:last_trial],
        )
        plt.xlim(0, 6000)
        fig.axes[0].set_title(
            f"Subject {subject_id}, trials {first_trial}-{last_trial}"
        )
        # plt.savefig(
        #     f"apnea-ecg/all_trials_plot/{subject_id}_trials_{first_trial}_{last_trial}.pdf"
        # )
        plt.show()
        plt.clf()
    print("done")
# %%
