import numpy as np

from alphacsc.utils.dictionary import (
    get_D_shape, _patch_reconstruction_error, get_uv
)


def get_z_nnz(z_hat):
    """

    Parameters
    ----------

    Returns
    -------
    """
    z_nnz = np.sum(z_hat != 0, axis=(0, 2))
    return z_nnz / z_nnz.shape[-1]


def get_max_error_patch(X, z, D):
    """
    Returns the patch of the signal with the largest reconstuction error.

    Parameters
    ----------
    X, z, D

    Returns
    -------
    D_k : ndarray, shape (n_channels, n_times_atom) or
            (n_channels + n_times_atom,)
        Patch of the residual with the largest error.
    """
    patch_rec_error = _patch_reconstruction_error(
        X, z, D
    )
    i0 = patch_rec_error.argmax()
    n0, t0 = np.unravel_index(i0, patch_rec_error.shape)

    n_channels = X.shape[1]
    *_, n_times_atom = get_D_shape(D, n_channels)

    patch = X[n0, :, t0:t0 + n_times_atom][None]
    if D.ndim == 2:
        patch = get_uv(patch)
    return patch
