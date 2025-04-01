import numpy as np
import torch

from rosecdl.utils.dictionary import _patch_reconstruction_error, get_D_shape, get_uv


def get_torch_generator(seed=None, device=None):
    """Return a torch.Generator object with the specified seed.

    Parameters
    ----------
    seed : int, optional
        Seed for the generator. If None, a random seed is used.
    device : str, optional
        Device to use for the generator.

    Returns
    -------
    torch.Generator
        Generator object for torch.

    """
    generator = torch.Generator(device=device)
    if seed is None:
        seed = np.random.randint(0, 2**32)
    generator.manual_seed(seed)
    return generator


def get_z_nnz(z_hat):
    """Calculate the number of non-zero elements across specified axes of z_hat array.

    Parameters
    ----------
    z_hat : numpy.ndarray
        Input array from which to count non-zero elements. Expected to be a 3D array.
    Returns
    -------
    numpy.ndarray
        Array containing counts of non-zero elements summed across axes 0 and 2,
        preserving the dimension of axis 1.

    Notes
    -----
    The function calculates element-wise count of non-zero values in z_hat
    across the first (0) and last (2) axes, effectively reducing the 3D array
    to a 1D array of counts.

    """
    z_nnz = np.sum(z_hat != 0, axis=(0, 2))
    # return z_nnz / z_nnz.shape[-1]
    return z_nnz


def get_max_error_patch(X, z, D):
    """Returns the patch of the signal with the largest reconstuction error.

    Parameters
    ----------
    X, z, D

    Returns
    -------
    D_k : ndarray, shape (n_channels, n_times_atom) or
            (n_channels + n_times_atom,)
        Patch of the residual with the largest error.

    """
    patch_rec_error = _patch_reconstruction_error(X, z, D)
    i0 = patch_rec_error.argmax()
    n0, t0 = np.unravel_index(i0, patch_rec_error.shape)

    n_channels = X.shape[1]
    *_, n_times_atom = get_D_shape(D, n_channels)

    patch = X[n0, :, t0 : t0 + n_times_atom][None]
    if D.ndim == 2:
        patch = get_uv(patch)
    return patch.copy()
