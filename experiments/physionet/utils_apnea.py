from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from alphacsc.learn_d_z import compute_X_and_objective, learn_d_z
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.update_z import update_z
from scipy.signal import tukey
from wfdb.io.annotation import rdann
from wfdb.io.record import rdrecord


def load_ecg(
    subject_id="a01", T=60, data_path=Path("apnea-ecg"), apply_window=True, verbose=True
):
    """Parameters
    ----------
    subject_id : str

    T : float
        duration, in seconds, of data splits
        default is 60 as the data have been annoted by the minute

    data_path : pathlib.Path
        path to data folder

    apply_window : bool (default: True)
        If set to True (default), a tukey window is applied to each split to
        reduce the border artifacts by reducing the weights of the chunk
        borders.

    verbose : bool
        if True, will print some information


    Returns
    -------
    X : ndarray, shape (n_splits, n_channels, int(T * fs))
        The signal splitted in ``n_splits``,
        whith ``n_splits = sig_len // int(T * fs)``,
        fs being the sampling frequency of the record

    labels : 1d array
        labels corresponding to one minute segments
        i.e., if T = 60, labels have the same length as data and each label
        corresponds to each datta split.

    """
    # ECG record
    record_name = str(data_path / subject_id)
    ecg_record = rdrecord(record_name=record_name)

    # split signal
    fs = ecg_record.fs  # sampling frequency of the record
    if verbose:
        print(f"Sampling frequency of the record: {fs} Hz")
    n_times = int(T * fs)
    n_splits = ecg_record.sig_len // n_times
    X = ecg_record.p_signal[: n_splits * n_times, :].T
    X = X.reshape(ecg_record.n_sig, n_splits, n_times).swapaxes(0, 1)

    # Apply a window to the signal to reduce the border artifacts
    if apply_window:
        X *= tukey(n_times, alpha=0.1)[None, None, :]

    # Add labels
    ann = rdann(
        record_name=record_name,
        extension="apn",
        return_label_elements=["symbol"],
        summarize_labels=True,
    )
    labels = np.array(ann.symbol)
    if T == 60:
        # ensure that labels and data have the same number of trials
        n_trials = min(len(ann.symbol), n_splits)
        return X[:n_trials], labels[:n_trials]
    else:
        import warnings

        warnings.warn(
            f"The returned labels do not match the data as T != 60 (got T = {T})."
        )
        return X, labels


def plot_records_sections(subject_id, n_sections=20, n_min_per_plot=5, first_section=0):
    """n_sections : int
        number of sections to plot

    n_min_per_plot : int
        number of minutes per plot to show

    first_section : int
        index of the first section to start from
    """
    data_path = Path("apnea-ecg")
    record_name = str(data_path / subject_id)
    ecg_record = rdrecord(record_name=record_name)
    ann = rdann(
        record_name=record_name,
        extension="apn",
        return_label_elements=["symbol"],
        summarize_labels=True,
    )

    for idx_min in (np.array(range(n_sections)) + first_section) * 5:
        wfdb.plot.plot_wfdb(
            ecg_record,
            ann,
            title=f"ECG-Apnea Record {subject_id}",
            return_fig=True,
            figsize=(50, 4),
        )

        plt.xlim(idx_min * 60, (idx_min + n_min_per_plot) * 60)
        plt.show()


def get_subject_info(subject_id):
    participants = pd.read_csv(Path("apnea-ecg/participants.tsv"), sep="\t")
    subject_info = participants[participants["Record"] == subject_id].iloc[0].to_dict()

    return subject_info


def plot_loss_history(pobj, times=None, save_fig=False):
    xx_type = "time"
    xlabel = "Time (s.)"

    if times is None:
        xx = np.arange(0, len(pobj) / 2, step=0.5)
        xx_type = "iteration"
        xlabel = "Iterations"
    elif np.any(np.diff(times) < 0):
        xx = np.cumsum(times)
    else:
        xx = times

    plt.plot(xx, pobj)
    plt.xlabel(xlabel)
    plt.xlim(0, None)
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.title(f"Loss history as function of {xx_type}")

    if save_fig:
        plt.savefig(save_fig / "loss_history.pdf")
    plt.show()
    plt.close()


def plot_temporal_atoms(d_hat, sfreq=100, save_fig=False):
    """Parameters
    ----------
    d_hat : array, shape (n_atoms, n_times)
        The estimated atoms.

    sfreq : float
        sampling frequency

    Returns
    -------

    """
    assert d_hat.ndim == 2

    n_atoms, n_times_atom = d_hat.shape
    t = np.arange(n_times_atom) / sfreq  # time support of the atom

    # define plot grid
    n_columns = min(5, n_atoms)
    n_split = int(np.ceil(n_atoms / n_columns))
    figsize = (4 * n_columns, 3 * n_split)
    fig, axes = plt.subplots(n_split, n_columns, figsize=figsize, sharey=True)

    if n_split == 1:
        axes = np.atleast_2d(axes)

    for ii, v_k in enumerate(d_hat):
        # Select the axes to display the current atom
        i_row, i_col = ii // n_columns, ii % n_columns
        ax = axes[i_row, i_col]

        # Plot the temporal pattern of the atom
        ax.set_title("Atom % d" % ii, pad=0)

        ax.plot(t, v_k)
        ax.set_xlim(min(t), max(t))
        if i_col == 0:
            ax.set_ylabel("Temporal")

    fig.tight_layout()
    if save_fig:
        plt.savefig(save_fig)
    plt.show()
    plt.close()


def plot_multi_subject_temporal_atoms(dict_d_hat, sfreq=100, save_fig=False):
    """Parameters
    ----------

    dict_d_hat : dict of arrays of shape (n_atoms, n_times)
        keys are strings
        values are fitted 2d-dictionaries

    Returns
    -------

    """
    n_atoms, n_times_atom = list(dict_d_hat.values())[0].shape
    t = np.arange(n_times_atom) / sfreq  # time support of the atom

    # define plot grid
    n_columns = n_atoms
    n_rows = len(dict_d_hat)
    figsize = (4 * n_columns, 3 * n_rows)
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=figsize, sharex=True, sharey=True
    )

    if n_rows == 1:
        axes = np.atleast_2d(axes)

    for ii, (subject_id, d_hat) in enumerate(dict_d_hat.items()):
        for kk, v_k in enumerate(d_hat):
            # Select the axes to display the current atom
            ax = axes[ii, kk]

            # Plot the temporal pattern of the atom
            ax.plot(t, v_k)
            ax.set_xlim(min(t), max(t))

            if ii == 0:
                ax.set_title("Atom % d" % kk, pad=0)

            if ii == (n_rows - 1):
                ax.set_xlabel("Time (s.)")

            if kk == 0:
                ax.set_ylabel(f"{subject_id} Temporal")

    fig.tight_layout()
    if save_fig:
        plt.savefig(save_fig)
    plt.show()
    plt.close()


def run_cdl(
    X,
    cdl_params,
    labels=None,
    fit_on="N",
    plot_loss=True,
    plot_atoms=True,
    save_fig=False,
):
    """Parameters
    ----------
    X : ndarray, shape (n_splits, n_channels, n_times)

    Returns
    -------

    """
    n_splits, n_channels, n_times = X.shape
    if n_channels == 1:
        X = X.squeeze()

    if labels is None:
        X_ = X
    elif fit_on is not None:
        X_ = X[labels == fit_on]

    if n_channels == 1:
        pobj, times, d_hat, z_hat, reg = learn_d_z(X_, **cdl_params)
    else:
        pobj, times, d_hat, z_hat, reg = learn_d_z_multi(X_, **cdl_params)

    if plot_loss:
        plot_loss_history(pobj, times, save_fig=save_fig)

    if plot_atoms:
        plot_temporal_atoms(d_hat, save_fig=save_fig)

    return pobj, times, d_hat, z_hat, reg


def get_subject_z_and_cost(subject_id, d_hat, label=None):
    X, labels = load_ecg(subject_id, verbose=False)
    if label is not None:
        X_ = X.squeeze()[labels == label].copy()
    else:
        X_ = X.squeeze().copy()
    X_ /= X_.std()
    z_hat = update_z(
        X_,
        d_hat,
        reg=0.1,
        solver="l-bfgs",
        solver_kwargs={"tol": 1e-4, "max_iter": 10_000},
    )
    cost = compute_X_and_objective(X_, z_hat, d_hat, reg=0.1)

    return z_hat, cost
