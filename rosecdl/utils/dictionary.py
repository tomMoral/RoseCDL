import numpy as np
from scipy.signal.windows import tukey


def flip_uv(uv, n_channels):
    """Ensure the temporal pattern v peak is positive for each atom.

    If necessary, multiply both u and v by -1.

    Parameter
    ---------
    uv: array, shape (n_atoms, n_channels + n_times_atom)
        Rank1 dictionary which should be modified.
    n_channels: int
        number of channels in the original multivariate series

    Return
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

    Return
    ------
    uv: array, shape (n_atoms, n_channels + n_times_atom)
    """
    n_atoms, n_channels, n_times_atom = D.shape
    uv = np.zeros((n_atoms, n_channels + n_times_atom))
    for k, d in enumerate(D):
        U, s, V = np.linalg.svd(d)
        uv[k] = np.r_[U[:, 0], V[0]]
    return flip_uv(uv, n_channels)


def tukey_window(n_times_atom):
    window = tukey(n_times_atom)
    window[0] = 1e-9
    window[-1] = 1e-9
    return window
    return window


def squeeze_all_except_one(X, axis=0):
    squeeze_axis = tuple(set(range(X.ndim)) - set([axis]))
    return X.squeeze(axis=squeeze_axis)


def prox_d(D, return_norm=False):
    norm_d = np.maximum(1, np.linalg.norm(D, axis=(1, 2), keepdims=True))
    D /= norm_d

    if return_norm:
        return D, squeeze_all_except_one(norm_d, axis=0)
    else:
        return D
