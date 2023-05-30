# %%
import os
import numpy as np
from pathlib import Path
import json

import matplotlib.pyplot as plt

from dicodile.utils.viz import display_dictionaries
from dripp.cdl.utils_plot import plot_z_boxplot

from utils_apnea import plot_loss_history

exp_paths = [Path(d.path) for d in os.scandir("./benchmark_apnea-ecg") if d.is_dir()]
# %%

for this_exp_path in exp_paths:
    with open(this_exp_path / "exp_params.json", "r") as f:
        exp_params = json.load(f)

    if exp_params["solver_z_kwargs"]["max_iter"] < 200:
        continue

    fit = exp_params["fit"]
    n_iter = exp_params["solver_z_kwargs"]["max_iter"]
    reg = exp_params["reg"]
    if n_iter < 1_000:
        continue

    print(this_exp_path)
    subjects_dirs = [d for d in this_exp_path.iterdir() if d.is_dir()]
    if len(subjects_dirs) < 5:
        continue

    d_hat_dict = {}
    losses = []
    times = []
    labels = []
    dict_z = {}
    for this_subject_dir in subjects_dirs:
        if (this_subject_dir / f"d_hat_{fit}.npy").exists():
            d_hat_dict[this_subject_dir.name] = np.load(
                this_subject_dir / f"d_hat_{fit}.npy"
            )
            losses.append(np.load(this_subject_dir / f"pobj_{fit}.npy"))
            times.append(np.load(this_subject_dir / f"times_{fit}.npy"))
            labels.append(this_subject_dir.name)
            # load z_hat and threshold
            z_hat = np.load((this_subject_dir / f"z_hat_{fit}.npy"))
            # get power of 10 of the max
            max_pow = int(np.floor(np.log10(abs(z_hat.max()))))
            threshold = 10 ** (max_pow - 5)  # restrict to 5 orders of magnitude
            z_hat[z_hat < threshold] = 0
            dict_z[this_subject_dir.name] = z_hat

    # plot all atoms
    list_D = [np.array(v) for v in d_hat_dict.values()]
    n_atoms = exp_params["n_atoms"]
    ncols = 5
    nrows = int(np.ceil(n_atoms / ncols)) * len(list_D)
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(3 * ncols, 2 * nrows),
    )
    fig = display_dictionaries(*list_D, axes=axes)
    plt.xlim(0, 75)
    plt.suptitle(f"Fit {fit}, {n_iter} iter, {reg} reg", y=0.925, fontsize=18)
    plt.savefig(this_exp_path / f"dictionaries_bench_{fit}_{n_iter}_{reg}.pdf")
    plt.show()
    plt.clf()

    # plot all losses
    plot_loss_history(
        losses,
        times=times,
        labels=labels,
        xscale="symlog",
        yscale="log",
        save_fig=this_exp_path / f"loss_history_bench_{fit}_{n_iter}_{reg}.pdf",
    )
    plt.clf()

    # plot z boxplot
    plot_z_boxplot(
        dict_z,
        type="box",
        add_number=True,
        fig_name=this_exp_path / f"z_boxplot_bench_{fit}_{n_iter}_{reg}.pdf",
    )
    plt.clf()


# %%
