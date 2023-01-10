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

from alphacsc.loss_and_gradient import _l2_gradient_d

from simulate import simulate_data

T = 10_000  # signal length
L = 100  # n times atom
W = 1_000  # window size
list_W = np.arange(100, T-L, 100)
n_times_valid = T - L + 1
n_acti_atom = 500


def compute_grad(X, z, D, i=None, W=1_000, extended=False):

    L = D.shape[-1]

    if i is not None:
        assert i <= (T-W), \
            f"i must be 0 <= i <= (T - W) = {T - W}, got i = {i}"

        if not extended:
            X = X[:, :, i:(i+W)].copy()  # shape (1, 1, W)
            z = z[:, :, i:(i+W-L+1)].copy()  # shape (1, 1, W-L+1)
        else:
            # pad with (L-1) zeros on both sides of the last dim
            i += L - 1

            X = np.pad(X, ((0, 0), (0, 0), (L-1, L-1)), constant_values=0)
            X = X[:, :, (i-L+1):(i+W+L-1)].copy()
            assert X.shape[-1] == (W + 2*L - 2), \
                f"last dim of X is {X.shape}, must be {W + 2*L - 2}"

            z = np.pad(z, ((0, 0), (0, 0), (L-1, L-1)), constant_values=0)
            z = z[:, :, (i-L+1):(i+W)].copy()
            assert z.shape[-1] == (W + L - 1), \
                f"last dim of z is {z.shape}, must be {W + L - 1}"

    _, grad = _l2_gradient_d(D, X, z)

    return grad


X, D, z = simulate_data(
    n_trials=1, n_channels=1, n_times=T, n_times_atom=L, n_atoms=1,
    n_acti_atom=n_acti_atom, random_state=42, constant_amplitude=False,
    window=True, shapes=['sin', 'gaussian'], sigma_noise=1, plot_atoms=False)

# compute full grad
full_grad = compute_grad(X, z, D)

dict_error = []

for W in list_W:
    # compute grad on window partition
    list_i_part = [i*W for i in range(T//W)]
    n_win = len(list_i_part)
    # get random indices
    list_i_rnd = np.random.choice((T-W-L), n_win, replace=False) + L

    for is_partition, this_list_i in zip([True, False], [list_i_part, list_i_rnd]):
        for is_extended in [True, False]:
            win_grad = [compute_grad(X, z, D, i, W, extended=is_extended)
                        for i in this_list_i]
            # compute error to full grad
            dict_error.extend([{
                'partition': is_partition, 'W': W, 'extended': is_extended,
                'error': norm(this_win_grad - full_grad)}
                for this_win_grad in win_grad])


df_err = pd.DataFrame(dict_error)

# sns.lineplot(data=df_err, x="W", y="error", hue="extended", style="partition")
sns.relplot(
    data=df_err, x="W", y="error",
    col="partition", hue="extended",
    kind="line"
)
plt.xscale('log')

# %%
plt.plot(list_W, [len(range(T//W)) for W in list_W])
plt.xlabel('W')
plt.xscale('log')
plt.xlim(min(list_W), None)
plt.title("Number of sub-windows in the partition")
# %%
