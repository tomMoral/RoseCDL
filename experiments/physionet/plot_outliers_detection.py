# %%
import numpy as np
import pandas as pd
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt

from utils_apnea import plot_multi_subject_temporal_atoms

df_res = pd.read_csv("apnea-ecg/df_b01_global.tsv", sep="\t")

# %%
root = Path("benchmark_apnea-ecg/b01")
exp_folders = [f for f in root.iterdir() if f.is_dir]


def get_exp_label_from_folder(exp_folder):
    with open(exp_folder / "exp_params.json", "r") as f:
        exp_params = json.load(f)

    exp_label = ""

    q = exp_params["q"]
    if q is not None:
        exp_label += f"q_{q}"

    l_freq = exp_params["l_freq"]
    h_freq = exp_params["h_freq"]
    if l_freq is not None:
        exp_label += "filtered"

    if exp_label == "":
        exp_label = "None"

    return exp_label, q, l_freq, h_freq

list_dict = []
for this_exp_folder in exp_folders:
    for seed in range(5):
        pobj = np.load(this_exp_folder / "b01" / f"pobj_all_{seed}.npy")
        times = np.load(this_exp_folder / "b01" / f"times_all_{seed}.npy")
        exp_label, q, l_freq, h_freq = get_exp_label_from_folder(this_exp_folder)
        i = 0
        for this_pobj, this_time in zip(pobj, times.cumsum()):
            data_dict = dict(
                q=q,
                l_freq=l_freq,
                h_freq=h_freq,
                exp_label=exp_label,
                pobj=this_pobj,
                pobj_norm=this_pobj - pobj[-1],
                times=this_time,
                seed=seed,
                iteration=i,
                compute_time=np.cumsum(times)[-1],
                n_iter=len(pobj),
            )
            list_dict.append(data_dict)
            i += 1

df = pd.DataFrame(list_dict)
%matplotlib inline

for y in ['pobj', 'pobj_norm']:
    for x in ['iteration', 'times']:
        sns.lineplot(
            data=df, x=x, y=y, hue="exp_label", style='seed'
        )
        plt.title(f"{y} as a function of {x}")
        plt.show()
        plt.clf()

df_ = df.drop_duplicates(
  subset = ['q', 'l_freq', 'seed'],
  keep = 'last').reset_index(drop = True)
sns.boxplot(data=df_, x="compute_time", y="exp_label")
plt.xscale('symlog')
plt.title("Computational time per CDL as function of type of pre-processing \n (5 points per box)")
plt.xlabel('Computational time per CDL fitting')
plt.ylabel('Type of pre-processing ')
plt.show()

sns.boxplot(data=df_, x="n_iter", y="exp_label")
plt.title("Number of iterations per CDL as function of type of pre-processing \n (5 points per box)")
plt.xlabel('Number of iterations per CDL fitting')
plt.ylabel('Type of pre-processing ')
plt.show()


# %% Plot some learned dictionaries

dict_d_hat = {}
for this_exp_folder in exp_folders:
    path = Path(f'{this_exp_folder}/b01')
    exp_label = get_exp_label_from_folder(this_exp_folder)[0]
    dict_d_hat[exp_label] = np.load(path / "d_hat_all_0.npy").squeeze()

plot_multi_subject_temporal_atoms(dict_d_hat)

# %%
