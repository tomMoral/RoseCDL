# %% Script to compute simularity between 2 dictionaries
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path


def recovery_score(D, D_ref, weights=None):
    """
    Compute a similarity score in [0, 1] between two dictionaries.

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_times_atoms + n_channels)
        Dictionary
    Dref : ndarray, shape (dim_signal, n_components)
        Reference dictionary
    weights : ndarray, shape (n_components)
        Weights of usage

    Returns
    -------
    score : float
        _Recovery score in [0, 1]
    """
    if weights is None:
        weights = np.ones(D.shape[0])

    cost_matrix = np.abs(D_ref @ ((D * weights[:, None]).T))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    score = cost_matrix[row_ind, col_ind].sum() / weights.sum()

    return score


reg = 0.1
lmbd_max = 'fixed'
suff = f"reg_{reg}_lmbd_{lmbd_max}"
# get 2 subjects' dictionary
subject_id_1 = 'sub-CC110033'
subject_dir = Path(f'../results_alphacsc/{subject_id_1}')
D_hat_1 = np.load(subject_dir / ('uv_hat_' + suff + '.npy'))

subject_id_2 = 'sub-CC110037'
subject_dir = Path(f'../results_alphacsc/{subject_id_2}')
D_hat_2 = np.load(subject_dir / ('uv_hat_' + suff + '.npy'))

# %%
recovery_score(D_hat_1, D_hat_2)
# %%

weights = np.abs(codes).sum(axis=(0, 2))
