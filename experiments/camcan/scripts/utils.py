from pathlib import Path

import numpy as np
from alphacsc.init_dict import init_dictionary
from alphacsc.loss_and_gradient import compute_X_and_objective_multi
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils.dictionary import get_lambda_max
from joblib import Parallel, delayed
from mne_bids import BIDSPath, read_raw_bids

from rosecdl.rosecdl import RoseCDL

DEVICE = "cuda:1"


def get_var_patch(X, n_times_atom, clip_value=None):
    patch = np.ones(shape=n_times_atom)

    var_patch = (
        np.sum([np.convolve(patch, diff_i, mode="valid") for diff_i in X**2], axis=0)
        / n_times_atom
    )
    var_patch -= (
        np.sum([np.convolve(patch, diff_i, mode="valid") for diff_i in X], axis=0)
        / n_times_atom
    ) ** 2

    if clip_value is not None:
        return var_patch.clip(clip_value)

    return var_patch


def get_subject_X(subject_path, n_times_atom=150, normalized=True, q=None):
    X = np.load(subject_path)

    if not normalized:
        return X

    # center
    X -= X.mean()
    # reduce
    var_patch = get_var_patch(X, n_times_atom, clip_value=0)
    if q is not None:
        var_patch = np.quantile(var_patch, q, overwrite_input=True)
    X /= np.sqrt(np.median(var_patch))
    return X


def get_lambda_global(
    list_subjects_path, n_atoms, n_times_atom, q=0.95, reg=0.3, method=np.median
):
    """For a list of subject, compute their lambda, and return a global value
    (median, mean, etc.)

    Parameters
    ----------

    list_subjects_path : list of paths

    method : callable

    reg : float


    Returns
    -------

    lambda_global : float

    list_lmbd : list
        list of all computed values of lambda_max, before applying reg and
        method

    """

    def proc(subject_path):
        # get a unique value for lambda
        # X = get_subject_X(subject_path, n_times_atom, q=q)
        X = np.load(subject_path)
        X -= X.mean(axis=1, keepdims=True)
        X /= X.std(axis=1, keepdims=True)
        # patch = np.ones(shape=n_times_atom)
        # var_patch = np.sum([np.convolve(patch, diff_i, mode='valid')
        #                     for diff_i in X**2], axis=0) / n_times_atom
        # var_patch -= (np.sum([np.convolve(patch, diff_i, mode='valid')
        #                      for diff_i in X], axis=0)/n_times_atom)**2
        # var_patch = var_patch.clip(0)

        # X /= np.sqrt(np.median(var_patch))

        # get initial dictionary with alphacsc
        D_init = init_dictionary(
            X[None, :],
            n_atoms,
            n_times_atom,
            uv_constraint="separate",
            rank1=True,
            window=True,
            D_init="chunk",
            random_state=None,
        )
        lmbd = get_lambda_max(X[None, :], D_init).max()
        return lmbd

    n_jobs = min(len(list_subjects_path), 40)
    list_lmbd = Parallel(n_jobs=n_jobs)(
        delayed(proc)(subject_path) for subject_path in list_subjects_path
    )

    lambda_global = reg * method(list_lmbd)

    return lambda_global, list_lmbd


def get_D_sub(subject_path, n_atoms=40, n_times_atom=150, lmbd=0.1):
    """Get subject's self dictionary using Windowing-CDL"""

    X = np.load(subject_path)
    X /= X.std()
    if X.ndim == 2:
        (n_channels, n_times) = X.shape
        n_trials = 1
    elif X.ndim == 3:
        (n_trials, n_channels, n_times) = X.shape

    # get initial dictionary with alphacsc
    D_init = init_dictionary(
        X[None, :],
        n_atoms,
        n_times_atom,
        uv_constraint="separate",
        rank1=True,
        window=True,
        D_init="chunk",
        random_state=None,
    )
    # lmbd = lmbd * get_lambda_max(X[None, :], D_init).max()

    CDL = RoseCDL(
        n_components=n_atoms,
        kernel_size=n_times_atom,
        n_channels=n_channels,
        lmbd=lmbd,
        n_iterations=30,
        epochs=50,
        max_batch=5,
        stochastic=False,
        optimizer="linesearch",
        lr=0.1,
        gamma=0.9,
        mini_batch_window=1_000,
        mini_batch_size=1,  # batch_size for the dataloader
        device=DEVICE,
        rank1=True,
        window=True,
        D_init=D_init,
        positive_z=True,
        list_D=False,
        dimN=1,
    )

    CDL.fit(X)

    return D_init, CDL.D_hat_, lmbd


def get_subject_z_and_cost(subject_path, uv_hat_, reg=0.1, tt_max=None):
    """Compute subject sparse vector for a given dictionnary and pre-processed
    signal.

    Parameters
    ----------
    subject_path : Pathlib instance
        path to subject's pre-processed raw signal as numpy array,
        must end by [subject_id].npy

    uv_hat_ : array-like, shape (n_atoms, n_channels + n_times_atom)
        learned atoms' dictionnary

    reg : float
        value for sparsity regularization

    tt_max : int | None
        if int, only the first `tt_max` indices will be considered
        if None, the whole signal is considered

    Returns
    -------
    subject_dict : dict
        subject_id, age, sex : subject infos
        z_hat : 2d-array shape (n_atoms, n_times)
        n_acti : 1d-array of length n_atoms
            number of non-null activations for each atom

    """

    subject_id = subject_path.name.split(".")[0]

    X = np.load(subject_path)
    X /= X.std()
    if tt_max is not None:
        X = X[:, :tt_max]
    # compute sparse vector z
    # for L0 update for z, pass reg to 0
    z_hat, _, _ = update_z_multi(
        X[None, :],
        uv_hat_.astype(np.float64),
        reg=reg,
        # XXX changed to 200_000
        solver="lgcd",
        solver_kwargs={"tol": 1e-4, "max_iter": 1_000_000},
        n_jobs=1,
    )
    # compute associated cost
    cost = compute_X_and_objective_multi(
        X[None, :], z_hat, D_hat=uv_hat_, reg=reg, uv_constraint="separate"
    )

    # normaliza with T
    cost /= X.shape[-1]

    return subject_id, z_hat, cost


def get_camcan_info(subject_id, return_raw=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    # paths to CamCAN files for Inria Saclay users
    DATA_DIR = Path("/storage/store/data/")
    BIDS_root = DATA_DIR / "camcan/BIDSsep/smt/"
    # SSS_CAL = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
    # CT_SPARSE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
    # PARTICIPANTS_FILE = BIDS_root / "participants.tsv"
    # get info
    # load_params = dict(sfreq=150.)
    # _, info = load_data_camcan(
    #     BIDS_root, SSS_CAL, CT_SPARSE, subject_id, **load_params)

    bp = BIDSPath(
        root=BIDS_root,
        subject=subject_id.split("-")[1],
        task="smt",
        datatype="meg",
        extension=".fif",
        session="smt",
    )
    raw = read_raw_bids(bp)
    raw.pick_types(meg="grad", eeg=False, eog=False, stim=False)
    info = raw.info

    if return_raw:
        return info, raw

    return info
