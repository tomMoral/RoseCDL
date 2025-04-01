import numpy as np
from scipy.signal.windows import tukey

from rosecdl.utils.validation import check_random_state


def _patch_reconstruction_error(X, z, D):
    """Return the reconstruction error for each patches of size (P, L)."""
    _, n_channels, _ = X.shape
    *_, n_times_atom = get_D_shape(D, n_channels)

    from rosecdl.convolution import construct_X_multi

    X_hat = construct_X_multi(z, D, n_channels=n_channels)

    diff = (X - X_hat) ** 2
    patch = np.ones(n_times_atom)

    return np.sum(
        [
            [np.convolve(patch, diff_ip, mode="valid") for diff_ip in diff_i]
            for diff_i in diff
        ],
        axis=1,
    )


def flip_uv(uv, n_channels):
    """Ensure the temporal pattern v peak is positive for each atom.

    If necessary, multiply both u and v by -1.

    Parameter
    ---------
    uv: array, shape (n_atoms, n_channels + n_times_atom)
        Rank1 dictionary which should be modified.
    n_channels: int
        number of channels in the original multivariate series

    Return:
    ------
    uv: array, shape (n_atoms, n_channels + n_times_atom)

    """
    v = uv[:, n_channels:]
    index_array = np.argmax(np.abs(v), axis=1)
    peak_value = v[np.arange(len(v)), index_array]
    uv[peak_value < 0] *= -1
    return uv


def get_uv(D):
    """Project D on the space of rank 1 dictionaries.

    Parameter
    ---------
    D: array, shape (n_atoms, n_channels, n_times_atom)

    Return:
    ------
    uv: array, shape (n_atoms, n_channels + n_times_atom)

    """
    n_atoms, n_channels, n_times_atom = D.shape
    uv = np.zeros((n_atoms, n_channels + n_times_atom))
    for k, d in enumerate(D):
        U, s, V = np.linalg.svd(d)
        uv[k] = np.r_[U[:, 0], V[0]]
    return flip_uv(uv, n_channels)


def prox_uv(uv, uv_constraint="joint", n_channels=None, return_norm=False):
    if uv_constraint == "joint":
        norm_uv = np.maximum(1, np.linalg.norm(uv, axis=1, keepdims=True))
        uv /= norm_uv

    elif uv_constraint == "separate":
        assert n_channels is not None
        norm_u = np.maximum(
            1, np.linalg.norm(uv[:, :n_channels], axis=1, keepdims=True)
        )
        norm_v = np.maximum(
            1, np.linalg.norm(uv[:, n_channels:], axis=1, keepdims=True)
        )

        uv[:, :n_channels] /= norm_u
        uv[:, n_channels:] /= norm_v
        norm_uv = norm_u * norm_v
    else:
        raise ValueError("Unknown uv_constraint: %s." % (uv_constraint,))

    if return_norm:
        return uv, squeeze_all_except_one(norm_uv, axis=0)
    return uv


def tukey_window(n_times_atom):
    window = tukey(n_times_atom)
    window[0] = 1e-9
    window[-1] = 1e-9
    return window


def tukey_window_2d(n_rows: int, n_cols: int):
    vertical_window = np.expand_dims(tukey(n_rows), axis=1)
    horizontal_window = np.expand_dims(tukey(n_cols), axis=0)
    window = vertical_window * horizontal_window
    window[0, :] = 1e-9
    window[-1, :] = 1e-9
    window[:, 0] = 1e-9
    window[:, -1] = 1e-9

    return window


def squeeze_all_except_one(X, axis=0):
    squeeze_axis = tuple(set(range(X.ndim)) - set([axis]))
    return X.squeeze(axis=squeeze_axis)


def prox_d(D, return_norm=False):
    norm_d = np.maximum(1, np.linalg.norm(D, axis=(1, 2), keepdims=True))
    D /= norm_d

    if return_norm:
        return D, squeeze_all_except_one(norm_d, axis=0)
    return D


def get_D_shape(D, n_channels):
    if D.ndim == 2:
        n_times_atom = D.shape[1] - n_channels
    else:
        if n_channels is None:
            n_channels = D.shape[1]
        else:
            assert (
                n_channels == D.shape[1]
            ), f"n_channels does not match D.shape: {D.shape}"
        n_times_atom = D.shape[2]

    return (D.shape[0], n_channels, n_times_atom)


def init_dictionary(
    X,
    n_atoms,
    n_times_atom,
    uv_constraint="separate",
    rank1=True,
    window=False,
    D_init=None,
    random_state=None,
):
    """Return an initial dictionary for the signals X.

    Parameter
    ---------
    X: array, shape(n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms: int
        The number of atoms to learn.
    n_times_atom: int
        The support of the atom.
    uv_constraint: str in {'joint' | 'separate'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
    rank1: boolean
        If set to True, use a rank 1 dictionary.
    window: boolean
        If True, multiply the atoms with a temporal Tukey window.
    D_init: array or {'chunk' | 'random'}
        The initialization scheme for the dictionary or the initial
        atoms. The shape should match the required dictionary shape, ie if
        rank1 is True, (n_atoms, n_channels + n_times_atom) and else
        (n_atoms, n_channels, n_times_atom)
    random_state: int | None
        The random state.

    Return:
    ------
    D: array shape(n_atoms, n_channels + n_times_atom) or
              shape(n_atoms, n_channels, n_times_atom)
        The initial atoms to learn from the data.

    """
    n_trials, n_channels, n_times = X.shape
    rng = check_random_state(random_state)

    D_shape = (n_atoms, n_channels, n_times_atom)
    if rank1:
        D_shape = (n_atoms, n_channels + n_times_atom)

    if isinstance(D_init, np.ndarray):
        D_hat = D_init.copy()
        assert D_hat.shape == D_shape

    elif D_init is None or D_init == "random":
        D_hat = rng.randn(*D_shape)

    elif D_init == "chunk":
        D_hat = np.zeros((n_atoms, n_channels, n_times_atom))
        for i_atom in range(n_atoms):
            i_trial = rng.randint(n_trials)
            t0 = rng.randint(n_times - n_times_atom)
            D_hat[i_atom] = X[i_trial, :, t0 : t0 + n_times_atom].copy()
        if rank1:
            D_hat = get_uv(D_hat)

    elif D_init == "greedy":
        raise NotImplementedError

    else:
        raise NotImplementedError(
            f"It is not possible to initialize uv with parameter {D_init}."
        )

    if window and not isinstance(D_init, np.ndarray):
        if rank1:
            D_hat[:, n_channels:] *= tukey_window(n_times_atom)[None, :]
        else:
            D_hat = D_hat * tukey_window(n_times_atom)[None, None, :]

    if rank1:
        D_hat = prox_uv(D_hat, uv_constraint=uv_constraint, n_channels=n_channels)
    else:
        D_hat = prox_d(D_hat)
    return D_hat
