# %%

import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from alphacsc.utils.convolution import construct_X_multi
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils.dictionary import get_lambda_max

from dripp.cdl.utils_plot import plot_z_boxplot

# from dicodile.utils.viz import display_dictionaries
# from dicodile.utils.utils import is_deacreasing

from utils_apnea import load_ecg, plot_subject_record, plot_1d_trials

# %%
# exp_id = "1f7ef48b9a0b6d977d6c48df5d610164"
# exp_id = "df2b41bf54803d98c4d8c3cca654a1e8"
# exp_id = "02709f34e2952b655894b95213a289ae"
# exp_id = "81a4f67accf8203c204f026169c649d4"  # n_iter = 1
# exp_id = "6625decca6f9af30c434fe2c8a796c16"  # n_iter = 10
# exp_id = "4c37255c701b5dfe64c505a4dc4398fd"  # n_iter = 100
# exp_id = "3710cfbd28f5ef5f924ad5b1340753fb"  # n_iter 500
# exp_id = "eec09f83266e9c63a8eccbc885211755"  # reg at 25 (A)
exp_id = "84c7fd9db9ce7e0164b4a2c7b1ed55ae"  # reg at 10 (A)
# exp_id = "a04676edc81a73c9288e61b400a9d59c" # reg at 10 (N)
# subject_id = "b05"

for i in [1, 2, 3, 4, 5]:
    subject_id = f"b0{i}"
    this_exp_path = Path("./benchmark_apnea-ecg") / exp_id
    this_subject_dir = this_exp_path / subject_id

    with open(this_exp_path / "exp_params.json", "r") as f:
        exp_params = json.load(f)

    fit = exp_params["fit"]

    with open(this_subject_dir / f"dict_res_{fit}.pkl", "rb") as f:
        dict_res = pickle.load(f)

    idx = dict_res["idx"]  # idexes of trials taken to fit CDL

    print(exp_params)

    # plot dict
    D_chunk = np.load((this_subject_dir / f"d_chunk_{fit}.npy"))
    D_hat = np.load((this_subject_dir / f"d_hat_{fit}.npy"))
    from dicodile.utils.viz import display_dictionaries

    %matplotlib inline
    display_dictionaries(D_chunk, D_hat)
    plt.savefig(f"{this_subject_dir}/{subject_id}_{exp_params['n_iter']}_dict.png")
    plt.show()
    plt.clf()
    mae = np.abs(D_chunk - D_hat).max()
    print(f"mae: {mae}")

    # plot loss
    pobj = np.load((this_subject_dir / f"pobj_{fit}.npy"))
    times = np.load((this_subject_dir / f"times_{fit}.npy"))
    plt.plot(times.cumsum(), pobj)
    plt.xlabel("Time (s.)")
    plt.yscale("log")
    plt.title(f"{subject_id} loss")
    plt.savefig(f"{this_subject_dir}/{subject_id}_{exp_params['n_iter']}_loss.png")
    plt.show()
    plt.clf()

    z_hat = np.load((this_subject_dir / f"z_hat_{fit}.npy"))
    restrict_z = False
    if restrict_z:
        # get power of 10 of the max
        max_pow = int(np.floor(np.log10(abs(z_hat.max()))))
        threshold = 10 ** (max_pow - 5)  # restrict to 5 orders of magnitude
        z_hat[z_hat < threshold] = 0

    idx_atoms = list(range(exp_params["n_atoms"]))
    shift = False
    if shift:
        # roll to put activation to the peak amplitude time in the atom.
        for kk in idx_atoms:
            shift = np.argmax(np.abs(D_hat[kk]))
            z_hat[:, kk] = np.roll(z_hat[:, kk], -shift)
            z_hat[:, kk, -shift:] = 0  # pad with 0

    X, labels = load_ecg(
        subject_id,
        split=True,
        T=60,
        apply_window=True,
        verbose=False,
    )
    # X = X[labels == fit]

    # for kk in idx_atoms:
    #     X_hat = construct_X_multi(
    #         z_hat[:, kk : kk + 1, :], D_hat[kk : kk + 1, :], n_channels=1
    #     )
    #     fig = plot_1d_trials(X[idx].squeeze(), X_hat.squeeze(), labels=idx)
    #     plt.show()
    X_hat = construct_X_multi(z_hat[:, :10, :], D_hat[:10, :], n_channels=1)
    fig = plot_1d_trials(X[idx].squeeze(), X_hat.squeeze(), labels=idx)
    plt.xlim(2000, 3000)
    fig.suptitle(
        f"Signal of subject {subject_id}, {exp_params['n_iter']} iterations",
        fontsize=14,
        y=0.9,
    )
    plt.savefig(f"{this_subject_dir}/{subject_id}_{exp_params['n_iter']}.png")
    plt.show()
    plt.clf()

    plot_z_boxplot(
        z_hat.swapaxes(0, 1),
        type="box",
        add_number=True,
        fig_name=this_subject_dir / f"z_boxplot_{fit}.pdf",
    )

1 / 0
# %%

z_hat_init, _, _ = update_z_multi(
    X[idx],
    D_chunk,
    reg=exp_params["reg"],
    solver=exp_params["solver_z"],
    solver_kwargs=exp_params["solver_z_kwargs"],
)
X_hat_init = construct_X_multi(z_hat_init, D_chunk, n_channels=1)


# %%


if idx == "all":
    idx = list(range(X_hat.shape[0]))
plot_subject_record(subject_id, fit, idx, X_hat=X_hat)
# Compute lambda max


lambda_max = get_lambda_max(X, D_hat, q=1)
lambda_max = np.quantile(lambda_max, q=1)
print("lambda_max:", lambda_max)

lambda_max_chunk = get_lambda_max(X, D_chunk, q=1)
lambda_max_chunk = np.quantile(lambda_max_chunk, q=1)
print("lambda_max_chunk:", lambda_max_chunk)
# %% recompute sparse code with learned dict
z_hat_rec, _, _ = update_z_multi(
    X[:20],
    D_hat,
    reg=exp_params["reg"],
    solver=exp_params["solver_z"],
    solver_kwargs=exp_params["solver_z_kwargs"],
)
X_hat_rec = construct_X_multi(z_hat_rec, D_hat, n_channels=1)

# %%
plot_subject_record(subject_id, fit, X_hat=X_hat_rec, start_trial=0, stop_trial=20)


# %%
