"""Evaluate WinCDL on one single subject of apnea-ecg, compared to alphacsc
results, by varying length of the window"""
# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from alphacsc.utils.signal import split_signal
from dicodile.utils.viz import display_dictionaries
from dripp.cdl.utils_plot import plot_z_boxplot

from wincdl.wincdl import WinCDL
from wincdl.datasets import PhysionetDataset, create_physionet_dataloader

from utils_apnea import load_ecg
from utils_apnea import plot_loss_history, plot_temporal_atoms

# load data
# subject_id = "a01"
# fit = "A"
group_id = "a"

subject_id_list = pd.read_csv(Path("apnea-ecg/participants.tsv"), sep="\t")[
    "Record"
].values
group_des = dict(a="apnea", b="borderline apnea", c="control", x="test")
subject_id_list = [id for id in subject_id_list if id[0] == group_id]

for subject_id in subject_id_list:
    print(f"Run for subject {subject_id}")
    for fit in ["A"]:  # , "N"]:
        print(f"Fit {fit}")
        if fit in ["A", "N"]:
            X, labels = load_ecg(
                subject_id, split=True, apply_window=True, verbose=False
            )
            X_ = X.squeeze()[labels == fit]
            n_trials, n_times = X_.shape
            X_ = X_.reshape(1, np.prod(X_.shape))
        elif fit == "all":
            X = load_ecg(subject_id, split=False, apply_window=False, verbose=False)
            labels = None
            n_times = X.shape[1]
            n_trials, n_times = 100, n_times // 100
            X_ = X[:, : n_trials * n_times]

        X_ /= X_.std()

        # %%
        subject_dir = Path(f"apnea-ecg/{subject_id}")
        D_init = np.load(subject_dir / f"d_init_{fit}.npy")
        D_init = D_init[:, None, :]

        apnea_cdl = WinCDL(
            n_components=3,
            kernel_size=75,
            n_channels=1,
            lmbd=0.1,  # learned alphacsc with fixed reg at 0.1
            n_iterations=100,  # Fista number of iterations
            epochs=100,
            max_batch=1,
            stochastic=False,
            optimizer="adam",
            lr=0.1,
            mini_batch_window=n_times,
            mini_batch_size=n_trials,
            overlap=False,
            device="cuda:1",
            rank="full",
            D_init=D_init,
            list_D=True,
            dimN=1,
        )
        losses, list_D, times = apnea_cdl.fit(X_, n_iter_eval=5_000)
        d_hat_win = apnea_cdl.D_hat_
        np.save(subject_dir / f"d_hat_win_{fit}", d_hat_win)
        # d_hat_win = apnea_cdl.D_hat_.squeeze()

        # %% load alphacsc results for this subject

        loss_alphacsc = np.load(subject_dir / f"pobj_{fit}.npy")
        times_alphacsc = np.load(subject_dir / f"times_{fit}.npy")
        d_hat_alphacsc = np.load(subject_dir / f"d_hat_{fit}.npy")
        z_hat_alphacsc = np.load(subject_dir / f"z_hat_{fit}.npy")

        # XXX : enlever le min des mins
        plot_loss_history(
            [loss_alphacsc, np.array(losses)],
            times=[times_alphacsc, np.array(times)],
            labels=["AlphaCSC", "WinCDL"],
            save_fig=subject_dir / f"loss_history_bench_{fit}.pdf",
        )
        plt.clf()

        # reshape
        # d_hat_win = d_hat_win.reshape(d_hat_win.shape[0], 1, d_hat_win.shape[1])
        d_hat_alphacsc = d_hat_alphacsc.reshape(
            d_hat_alphacsc.shape[0], 1, d_hat_alphacsc.shape[1]
        )

        ncols = 4
        nrows = 2
        fig, axes = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            squeeze=False,
            sharex=True,
            sharey=True,
            figsize=(3 * ncols, 2 * nrows),
        )
        fig = display_dictionaries(d_hat_alphacsc, d_hat_win, axes=axes)
        plt.xlim(0, 75)
        plt.savefig(subject_dir / f"dictionaries_bench_{fit}.pdf")
        plt.show()
        plt.clf()

        dict_z = dict(
            AlphaCSC=z_hat_alphacsc.swapaxes(0, 1),
            WinCDL=apnea_cdl.csc.z.to("cpu").detach().numpy(),
        )
        plot_z_boxplot(
            dict_z,
            type="box",
            add_number=True,
            fig_name=subject_dir / f"z_boxplot_bench_{fit}.pdf",
        )
        plt.clf()

# %%
