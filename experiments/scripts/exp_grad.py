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
# W = 1_000  # window size
list_W = np.arange(2*L, T-L, L)
n_times_valid = T - L + 1
p_acti = 1  # average number of activations per window of size L
n_acti_atom = int(T/L * p_acti)


def compute_grad(X, z, D, i=None, W=1_000, extended=False):
    """Compute the gradient of the reconstruction loss relative to d.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times) or None
        The data array
    z : array, shape (n_atoms, n_trials, n_times_valid) or None
        The activations
    D : array, shape
        (n_atoms, n_channels, n_times_atom)

    Returns
    -------
    grad : array, shape (n_atoms * n_times_valid)
        The gradient
    """

    T = X.shape[-1]
    L = D.shape[-1]

    if extended:  # pad with (L-1) zeros on both sides of the last dim
        X = np.pad(X, ((0, 0), (0, 0), (L-1, L-1)), constant_values=0)
        z = np.pad(z, ((0, 0), (0, 0), (L-1, L-1)), constant_values=0)

    if i is not None:
        assert i <= (T-W), \
            f"i must be 0 <= i <= (T - W) = {T - W}, got i = {i}"

        if not extended:
            X = X[:, :, i:(i+W)].copy()      # shape (n_trials, n_channels, W)
            z = z[:, :, i:(i+W-L+1)].copy()  # shape (n_atoms, n_trials, W-L+1)
        else:
            i += L - 1
            # shape (n_trials, n_channels, W+2*L-2)
            X = X[:, :, (i-L+1):(i+W+L-1)].copy()
            # shape (n_atoms, n_trials, W+L-1)
            z = z[:, :, (i-L+1):(i+W)].copy()

    _, grad = _l2_gradient_d(D, X, z)

    return grad


X, D, z = simulate_data(
    n_trials=1, n_channels=1, n_times=T, n_times_atom=L, n_atoms=1,
    n_acti_atom=n_acti_atom, random_state=42, constant_amplitude=False,
    window=True, shapes=['sin', 'gaussian'], sigma_noise=1, plot_atoms=False)

# compute full grad

dict_error = []
dict_esp = []

for L in [50, 100, 500]:
    list_W = np.arange(2*L, T-L, L)

    for p_acti in [1, 5, 10]:
        n_acti_atom = int(T/L * p_acti)

        X, D, z = simulate_data(
            n_trials=1, n_channels=1, n_times=T, n_times_atom=L, n_atoms=1,
            n_acti_atom=n_acti_atom, random_state=42, constant_amplitude=False,
            window=True, shapes=['sin', 'gaussian'], sigma_noise=1, plot_atoms=False)

        for W in list_W:
            # compute grad on window partition
            list_i_part = [i*W for i in range(T//W)]
            n_win = len(list_i_part)
            # get random indices
            list_i_rnd = np.random.choice((T-W-L), 20, replace=True) + L

            for is_extended in [True, False]:
                full_grad = compute_grad(X, z, D, extended=is_extended)
                rel_full_grad = full_grad / (T + is_extended*2*(L-1))

                for is_partition, this_list_i in \
                        zip([True, False], [list_i_part, list_i_rnd]):

                    win_grad = np.array([compute_grad(
                        X, z, D, i, W, extended=is_extended)
                        for i in this_list_i])
                    # compute error to full grad
                    dict_error.extend([{
                        'L': L, 'p_acti': p_acti, 'partition': is_partition,
                        'W': W, 'extended': is_extended,
                        'error': norm(this_win_grad/(W + is_extended*2*(L - 1)) - rel_full_grad)}
                        for this_win_grad in win_grad])

df_err = pd.DataFrame(dict_error)
df_err.to_csv('../results/df_exp_grad')

# %%
# Varying p_acti and partitioning (fixed L)
L = 100
g = sns.relplot(
    data=df_err[(df_err['L'] == L) & (df_err['W'] >= min(list_W))],
    x="W", y="error",
    col="partition", row="p_acti",
    hue="extended",
    kind="line"
)
(g.set_axis_labels('W', 'L2 error norm')
  .tight_layout()
 )
plt.xscale('log')
plt.xlim(min(list_W), None)
plt.subplots_adjust(top=0.9)
plt.suptitle(
    f'L2 norm between sub-window gradient and full signal one \n(T={T:.1e}, L={L})')
plt.savefig('../figures/exp_grad_partition_pacti.pdf')
plt.show()


# Varying L and p_acti (partition or random)
partition = False
g = sns.relplot(
    data=df_err[(df_err['partition'] == partition)
                & (df_err['W'] >= min(list_W))],
    x="W", y="error",
    col="L", row="p_acti",
    hue="extended",
    kind="line"
)
(g.set_axis_labels('W', 'L2 error norm')
  .tight_layout()
 )
plt.xscale('log')
plt.xlim(min(list_W), None)
plt.subplots_adjust(top=0.9)
plt.suptitle(
    f"L2 norm between {'partition' if partition else 'random'} sub-window gradient and full signal one \n(T={T:.1e})")
plt.savefig('../figures/exp_grad_L_pacti.pdf')
plt.show()

# %%
