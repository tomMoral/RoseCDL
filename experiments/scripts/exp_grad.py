"""
Experiment to compute the sub gradient on true sub-window and extended one over
sparse code vector, and compre it to true gradient computed over full signal.
"""
# %%
import numpy as np
import pandas as pd
from scipy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt

from simulate import simulate_data

from alphacsc.utils import check_random_state, construct_X, construct_X_multi
from alphacsc.utils.dictionary import tukey_window
from alphacsc.loss_and_gradient import _dense_transpose_convolve_z, _l2_gradient_d

T = 10_000  # signal length
L = 100  # n times atom
W = 1_000  # window size
n_times_valid = T - L + 1
n_acti_atom = 500

# simulate sin atom
# D = np.array([np.sin(2 * np.pi * np.linspace(0, 1, L))])
# D /= np.linalg.norm(D, axis=1)[:, None]
# if window:
#     D = D * tukey_window(L)[None, :]


# simulate sparse activation vector


# %%


def compute_grad(X, z, D, i=None, W=1_000, extended=False):

    L = D.shape[-1]

    if i is not None:
        assert i <= (T-W), \
            f"i must be 0 <= i <= (T - W) = {T - W}, got i = {i}"

        if not extended:
            X = X[:, :, i:(i+W)].copy()  # shape (1, 1, W)
            z = z[:, :, i:(i+W-L+1)].copy()  # shape (1, 1, W-L+1)
        else:
            i += L - 1
            # pad with (L-1) zeros on both sides of the last dim
            X = np.pad(X, ((0, 0), (0, 0), (L-1, L-1)), constant_values=0)
            X = X[:, :, (i-L+1):(i+W+L-1)].copy()
            assert X.shape[-1] == (W + 2*L - 2), f"X shape is {X.shape}"
            z = np.pad(z, ((0, 0), (0, 0), (L-1, L-1)), constant_values=0)
            z = z[:, :, (i-L+1):(i+W)].copy()
            assert z.shape[-1] == (W + L - 1), f"z shape is {z.shape}"

    _, grad = _l2_gradient_d(D, X, z)

    return grad


X, D, z = simulate_data(
    n_trials=1, n_channels=1, n_times=T, n_times_atom=L, n_atoms=1,
    n_acti_atom=n_acti_atom, random_state=42, constant_amplitude=False,
    window=True, shapes=['sin', 'gaussian'], sigma_noise=1, plot_atoms=False)

# %%

# compute full grad
full_grad = compute_grad(X, z, D)

dict_error = []

for W in np.arange(100, T+100, 100):
    # compute grad on window partition
    list_i = [i*W for i in range(T//W)]
    n_win = len(list_i)
    win_grad = [compute_grad(X, z, D, i, W, extended=False) for i in list_i]
    ext_grad = [compute_grad(X, z, D, i, W, extended=True) for i in list_i]

    # compute error to full grad
    dict_error.extend([{'partition': True, 'W': W, 'extended': False,
                        'error': norm(this_win_grad - full_grad/n_win)}
                       for this_win_grad in win_grad])
    dict_error.extend([{'partition': True, 'W': W, 'extended': True,
                        'error': norm(this_ext_grad - full_grad/n_win)}
                       for this_ext_grad in ext_grad])

df_err = pd.DataFrame(dict_error)

sns.lineplot(data=df_err, x="W", y="error", hue="extended")
plt.xscale('log')
# %%
