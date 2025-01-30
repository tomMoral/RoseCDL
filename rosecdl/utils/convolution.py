import numpy as np

from rosecdl.utils.dictionary import get_D_shape


def _sparse_convolve(z_i, ds):
    """Perform sparse convolution."""
    n_atoms, n_times_atom = ds.shape
    n_atoms, n_times_valid = z_i.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros(n_times)
    for zik, dk in zip(z_i, ds):
        for nnz in np.where(zik != 0)[0]:
            Xi[nnz : nnz + n_times_atom] += zik[nnz] * dk
    return Xi


def _dense_convolve(z_i, ds):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
    return sum([np.convolve(zik, dk) for zik, dk in zip(z_i, ds)], 0)


def _choose_convolve(z_i, ds):
    """Choose between _dense_convolve and _sparse_convolve.

    Use a heuristic on the sparsity of z_i, and perform the convolution.

    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    ds : array, shape(n_atoms, n_times_atom)
        Dictionary
    """
    assert z_i.shape[0] == ds.shape[0]

    if np.sum(z_i != 0) < 0.01 * z_i.size:
        return _sparse_convolve(z_i, ds)
    else:
        return _dense_convolve(z_i, ds)


def _sparse_convolve_multi(z_i, ds):
    """Sparse convolution multi."""
    n_atoms, n_channels, n_times_atom = ds.shape
    n_atoms, n_times_valid = z_i.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros(shape=(n_channels, n_times))
    for zik, dk in zip(z_i, ds):
        for nnz in np.where(zik != 0)[0]:
            Xi[:, nnz : nnz + n_times_atom] += zik[nnz] * dk
    return Xi


def _sparse_convolve_multi_uv(z_i, uv, n_channels):
    """Sparse convolution uv constraint."""
    u = uv[:, :n_channels]
    v = uv[:, n_channels:]
    n_atoms, n_times_valid = z_i.shape
    n_atoms, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1

    Xi = np.zeros(shape=(n_channels, n_times))
    for zik, uk, vk in zip(z_i, u, v):
        zik_vk = np.zeros(n_times)
        for nnz in np.where(zik != 0)[0]:
            zik_vk[nnz : nnz + n_times_atom] += zik[nnz] * vk

        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _dense_convolve_multi(z_i, ds):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
    return np.sum(
        [[np.convolve(zik, dkp) for dkp in dk] for zik, dk in zip(z_i, ds)], 0
    )


def _dense_convolve_multi_uv(z_i, uv, n_channels):
    """Convolve z_i[k] and uv[k] for each atom k, and return the sum."""
    u = uv[:, :n_channels]
    v = uv[:, n_channels:]
    n_atoms, n_times_valid = z_i.shape
    n_atoms, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1

    Xi = np.zeros((n_channels, n_times))
    for zik, uk, vk in zip(z_i, u, v):
        zik_vk = np.convolve(zik, vk)
        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _choose_convolve_multi(z_i, D=None, n_channels=None):
    """Choose between _dense_convolve and _sparse_convolve.

    Use a heuristic on the sparsity of z_i, and perform the convolution.

    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    n_channels : int
        Number of channels
    """
    assert z_i.shape[0] == D.shape[0]

    if np.sum(z_i != 0) < 0.01 * z_i.size:
        if D.ndim == 2:
            return _sparse_convolve_multi_uv(z_i, D, n_channels)
        else:
            return _sparse_convolve_multi(z_i, D)

    else:
        if D.ndim == 2:
            return _dense_convolve_multi_uv(z_i, D, n_channels)
        else:
            return _dense_convolve_multi(z_i, D)


def construct_X(z, ds):
    """Construct signal from dictionary and activation vector.

    Parameters
    ----------
    z : array, shape (n_atoms, n_trials, n_times_valid)
        The activations
    ds : array, shape (n_atoms, n_times_atom)
        The atoms

    Returns
    -------
    X : array, shape (n_trials, n_times)
    """
    assert z.shape[0] == ds.shape[0]
    n_atoms, n_trials, n_times_valid = z.shape
    n_atoms, n_times_atom = ds.shape
    n_times = n_times_valid + n_times_atom - 1

    X = np.zeros((n_trials, n_times))
    for i in range(n_trials):
        X[i] = _choose_convolve(z[:, i], ds)
    return X


def construct_X_multi(z, D=None, n_channels=None):
    """Construct X multi channels.

    Parameters
    ----------
    z : array, shape (n_trials, n_atoms, n_times_valid)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The activations
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    n_channels : int
        Number of channels

    Returns
    -------
    X : array, shape (n_trials, n_channels, n_times)
    """
    n_trials, n_atoms, n_times_valid = z.shape
    assert n_atoms == D.shape[0]
    _, n_channels, n_times_atom = get_D_shape(D, n_channels)
    n_times = n_times_valid + n_times_atom - 1

    X = np.zeros((n_trials, n_channels, n_times))
    for i in range(n_trials):
        X[i] = _choose_convolve_multi(z[i], D=D, n_channels=n_channels)
    return X
