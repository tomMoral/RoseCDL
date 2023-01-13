"""
Experiment to compute the sub gradient on true sub-window and extended one over
sparse code vector, and compare it to true gradient computed over full signal.
"""
# %%
import numpy as np
import pandas as pd
from scipy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt

from alphacsc.loss_and_gradient import _l2_gradient_d, _dense_transpose_convolve_z
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils import construct_X_multi

from simulate import simulate_data

T = 10_000  # signal length
# default values
L = 100  # n times atom
W = 1_000  # window size
list_W = np.arange(2*L, T-L, L)
n_times_valid = T - L + 1
p_acti = 1  # average number of activations per window of size L
n_acti_atom = int(T/L * p_acti)

list_L = [50, 100, 500]
list_p_acti = [1, 5, 10]
# %%


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
    pad = L - 1

    n_trials, n_channels, n_times = X.shape

    if i is None:  # compute full gradient
        _, grad = _l2_gradient_d(D, X, z, constants=None)
        return grad

    # get window from X and z
    X_w = X[:, :, i:(i+W)].copy()  # shape (n_trials, n_channels, W)
    z_w = z[:, :, i:(i+W-L+1)].copy()  # shape (n_atoms, n_trials, W-L+1)
    # pad with (L-1) zeros on both sides of the last dim
    z_pad = np.pad(z, ((0, 0), (0, 0), (pad, pad)), constant_values=0)
    i += pad
    # shape (n_atoms, n_trials, W+L-1)
    z_w_ext = z_pad[:, :, (i-pad):(i+W)].copy()
    # shape (n_trials, n_channels, W+2*L-2)
    X_hat = construct_X_multi(z_w_ext, D=D, n_channels=n_channels)
    X_hat = X_hat[:, :, L-1:L-1+W]  # shape (n_trials, n_channels, W)
    residual = X_w - X_hat
    grad = _dense_transpose_convolve_z(residual, z_w)
    return grad


X, D, z = simulate_data(
    n_trials=1, n_channels=1, n_times=T, n_times_atom=L, n_atoms=1,
    n_acti_atom=n_acti_atom, random_state=42, constant_amplitude=False,
    window=True, shapes=['sin', 'gaussian'], sigma_noise=1, plot_atoms=False)

n_trials, n_channels, n_times = X.shape


def test_convolve(z, D, W, i=0):

    L = D.shape[-1]

    X_hat = construct_X_multi(z, D=D, n_channels=n_channels)
    X_hat = X_hat[:, :, i:i+W]
    # pad
    z_pad = np.pad(z, ((0, 0), (0, 0), (L-1, L-1)), constant_values=0)
    i += L-1
    z_w_ext = z_pad[:, :, (i-L+1):(i+W)].copy()
    X_hat_ext = construct_X_multi(z_w_ext, D, n_channels=n_channels)
    X_hat_ = X_hat_ext[:, :, L-1:L-1+W].copy()
    residual = X_hat - X_hat_

    return norm(residual)


# compute full grad
full_grad = compute_grad(X, z, D, i=None)
# compute list of windowed grads
W = int(T/10)

win_grads = np.array([compute_grad(X, z, D, i=this_i*W, W=W)
                      for this_i in range(T//W)])
print(f'norm of diff: {norm(win_grads.mean(axis=0)/W - full_grad/T)}')


# %%
T = 10_000
W = 1_000
list_W = np.arange(W, T+W, W)
list_L = [50, 100, 500]
list_p_acti = [1, 5, 10]

dict_error = []
dict_esp = []

for L in list_L:
    for p_acti in list_p_acti:
        n_acti_atom = int(T/L * p_acti)

        X, D, z = simulate_data(
            n_trials=1, n_channels=1, n_times=T, n_times_atom=L, n_atoms=1,
            n_acti_atom=n_acti_atom, random_state=42, constant_amplitude=False,
            window=True, shapes=['sin', 'gaussian'], sigma_noise=1, plot_atoms=False)

        full_grad = compute_grad(X, z, D)

        for W in list_W:
            # compute grad on window partition
            list_i_part = [i*W for i in range(T//W)]
            n_win = len(list_i_part)

            win_grad = np.array([compute_grad(X, z, D, i, W)
                                 for i in list_i_part])
            dict_esp.append({
                'L': L, 'p_acti': p_acti, 'W': W,
                'error_esp': norm(win_grad.mean(axis=0)/W - full_grad/T)}
            )
            print(f"norm of diff: {dict_esp[-1]['error_esp']}")

            # get 20 random indices
            # list_i_rnd = np.random.choice((T-W-L), 20, replace=True) + L

            # for is_extended in [False]:
            #     full_grad = compute_grad(X, z, D, extended=is_extended)
            #     rel_full_grad = full_grad / (T + is_extended * 2 * (L-1))

            # for is_partition, this_list_i in \
            #         zip([True], [list_i_part]):

            #     win_grad = np.array([compute_grad(
            #         X, z, D, i, W, extended=is_extended)
            #         for i in this_list_i])
            #     # compute error to full grad
            #     dict_error.extend([{
            #         'L': L, 'p_acti': p_acti, 'partition': is_partition,
            #         'W': W, 'extended': is_extended,
            #         'error': norm(this_win_grad/(W + is_extended*2*(L - 1)) - rel_full_grad)}
            #         for this_win_grad in win_grad])
            #     dict_esp.append({
            #         'L': L, 'p_acti': p_acti, 'partition': is_partition,
            #         'W': W, 'extended': is_extended,
            #         'error_esp': norm(win_grad.sum(axis=0) - full_grad)}
            #     )

# df_err = pd.DataFrame(dict_error)
# df_err.to_csv('../results/df_exp_grad')
df_esp = pd.DataFrame(dict_esp)
sns.relplot(
    data=df_esp,
    x="W", y="error_esp",
    col="L", row="p_acti",
    kind="line"
)
plt.show()

# %%
# Varying p_acti and partitioning (fixed L)
# L = 100
# g = sns.relplot(
#     data=df_err[(df_err['L'] == L) & (df_err['W'] >= min(list_W))],
#     x="W", y="error",
#     col="partition", row="p_acti",
#     hue="extended",
#     kind="line"
# )
# (g.set_axis_labels('W', 'L2 error norm')
#   .tight_layout()
#  )
# plt.xscale('log')
# plt.xlim(min(list_W), None)
# plt.subplots_adjust(top=0.9)
# plt.suptitle(
#     f'L2 norm between sub-window gradient and full signal one \n(T={T:.1e}, L={L})')
# plt.savefig('../figures/exp_grad_partition_pacti.pdf')
# plt.show()


# # Varying L and p_acti (partition or random)
# partition = True
# g = sns.relplot(
#     data=df_err[(df_err['partition'] == partition)
#                 & (df_err['W'] >= min(list_W))],
#     x="W", y="error",
#     col="L", row="p_acti",
#     hue="extended",
#     kind="line"
# )
# (g.set_axis_labels('W', 'L2 error norm')
#   .tight_layout()
#  )
# plt.xscale('log')
# plt.xlim(min(list_W), None)
# plt.subplots_adjust(top=0.9)
# plt.suptitle(
#     f"L2 norm between {'partition' if partition else 'random'} sub-window gradient and full signal one \n(T={T:.1e})")
# plt.savefig('../figures/exp_grad_L_pacti.pdf')
# plt.show()

# Varying L and p_acti (partition or random)
partition = True
g = sns.relplot(
    data=df_esp[(df_esp['partition'] == partition)
                & (df_esp['W'] >= min(list_W))],
    x="W", y="error_esp",
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
