import numpy as np


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
