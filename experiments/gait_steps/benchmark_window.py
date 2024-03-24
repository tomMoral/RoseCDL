"""Benchmark to quantify the ability of the sub-windowing to recover 
"""
# %%
import numpy as np
from alphacsc.init_dict import init_dictionary
from alphacsc.utils.convolution import sort_atoms_by_explained_variances
from dicodile.utils.viz import display_dictionaries
import matplotlib.pyplot as plt

from wincdl.wincdl import WinCDL
from utils_gait import get_gait_data, get_participants

WINDOW = True
N_ATOMS = 8
N_TIMES_ATOM = 200
sort_atoms = False

participants = get_participants()
subjects_healthy = participants[participants["PathologyGroup"] == "Healthy"][
    "Subject"
].values


subject = subjects_healthy[0]

trial = 1
X = get_gait_data(subject, trial, verbose=False)["data"]["RAV"].to_numpy()
X /= X.std()
# reshape X to (n_trials, n_channels, n_times)
X_csc = X.reshape(1, 1, *X.shape)
X_win = X.reshape(1, *X.shape)
n_channels, n_times = X_win.shape

# %%
D_init = init_dictionary(
    X_csc,
    n_atoms=N_ATOMS,
    n_times_atom=N_TIMES_ATOM,
    rank1=False,
    window=WINDOW,
    D_init="chunk",
    random_state=None,
)

dict_loss = {}
dict_times = {}
dict_D = {}
dict_z = {}

for params in [(1, True), (0.5, True), (0.2, True), (0.2, False)]:
    p, overlap = params
    mini_batch_window = int(n_times * p)
    gait_cdl = WinCDL(
        n_components=N_ATOMS,
        kernel_size=N_TIMES_ATOM,
        n_channels=n_channels,
        # lmbd=dicod_cdl.reg_,
        lmbd=1.5134614391157055,
        n_iterations=1_000,  # Fista iterations for z step
        epochs=5,
        max_batch=int(1 / p),
        stochastic=True,  # if false, fit on full signal
        optimizer="linesearch",
        lr=0.1,
        mini_batch_window=mini_batch_window,
        overlap=overlap,  # allow overlap between windows
        mini_batch_size=1,
        device="cuda:1",
        rank="full",
        window=True,
        D_init=D_init,
        list_D=True,
        dimN=1,
    )
    losses, list_D, times = gait_cdl.fit(X_win, n_iter_eval=5_000)
    dict_loss[params] = np.array(losses)
    dict_times[params] = np.array(times)

    # sort_atoms = True
    if sort_atoms:
        D_hat_torch, z_hat_torch = sort_atoms_by_explained_variances(
            list_D[-1], gait_cdl.csc.z.cpu().numpy(), n_channels=n_channels
        )
    else:
        D_hat_torch = list_D[-1]
        z_hat_torch = gait_cdl.csc.z.cpu().numpy()

    dict_D[params] = D_hat_torch
    dict_z[params] = z_hat_torch


# %%
keys = [(1, True), (0.5, True), (0.2, True), (0.2, False)]

ncols = 4
nrows = (1 + len(keys)) * 2
fig, axes = plt.subplots(
    ncols=ncols,
    nrows=nrows,
    squeeze=False,
    sharex=True,
    sharey=True,
    figsize=(3 * ncols, 2 * nrows),
)
fig = display_dictionaries(D_init, *dict_D.values(), axes=axes)
plt.tight_layout()
plt.savefig("win_atoms.pdf")
plt.show()
plt.clf()

plt.plot(-np.ones(1))
for p in keys:
    plt.plot(dict_loss[p], label=p, alpha=0.8)
plt.yscale("log")
plt.xlabel("Times (s.)")
plt.ylabel("Lasso loss")
plt.legend()
plt.savefig("win_loss.pdf")
plt.show()
plt.clf()

# plot z_nnz
xx = np.array(range(N_ATOMS))
plt.plot(xx, -5 * np.ones(xx.shape))
for p in keys:
    plt.plot(xx, np.count_nonzero(dict_z[p], axis=(0, 2)), label=p)
plt.xlabel("Atom id")
plt.ylabel("# non-zero activations")
plt.xlim(0, N_ATOMS - 1)
plt.ylim(0, None)
plt.legend()
plt.savefig("win_acti.pdf")
plt.show()
plt.clf()

# %%
