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

from alphacsc.loss_and_gradient import _dense_transpose_convolve_z
from alphacsc.utils import construct_X_multi

from simulate import simulate_data

#
T = 1000
L = 10
W = 100
z = np.random.randn(T)
z = np.pad(z, (L-1, L-1))
d = np.random.randn(L)
X = np.convolve(z, d)
X += .1 * np.random.randn(*X.shape)
print(f'X shape: {X.shape}, T+2*(L-1) + L - 1: {T+2*(L-1) + L - 1}')

X_hat = np.convolve(z, d)

n = W - 2 * (L-1)
assert n > 0
weight = 1 / np.r_[np.ones(L-1), np.arange(n) + 1, n *
                   np.ones(T - 2*n), np.arange(n, 0, -1), np.ones(L-1)]
plt.plot(weight)
plt.title("weight")
plt.show()
# weight is same dim as z
grad = np.correlate(z, X_hat - X, mode='valid')
print(f'grad shape: {grad.shape}')

g = np.zeros_like(grad)
slice_win = slice(L-1, -L+1)  # [9:-9]
for i in range(z.shape[0] - W + 1):
    w = slice(i, i+W)
    w_X = slice(w.start, w.stop + L - 1)
    z_win = z[w]    # z[i:i+W]    length = W
    X_win = X[w_X]  # X[i:i + W+L-1]  length = W + L - 1
    weight_win = weight[w][slice_win]  # weight[i+L-1 : i+W-L+1]

    X_hat_win = np.convolve(z_win, d)
    assert (X_hat_win - X_win)[slice_win].all()
    grad_win = np.correlate(
        z_win[slice_win] * weight_win, (X_hat_win - X[w_X])[slice_win], mode='valid')
    grad_win = np.correlate(
        z_win[slice_win], (X_hat_win - X[w_X])[slice_win], mode='valid')
    residual = (X_hat_win - X[w_X])
    # grad_win = np.convolve(residual, z_win, mode='valid')
    g += grad_win

abs(g - grad).sum()


# %%

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

    n_trials, n_channels, n_times = X.shape

    if i is None:  # compute full gradient
        # _, grad = _l2_gradient_d(D, X, z, constants=None)
        X_hat = construct_X_multi(z, D=D, n_channels=n_channels)
        residual = X - X_hat
        grad = _dense_transpose_convolve_z(residual, z)
        return grad

    # get window from X and z
    X_w = X[:, :, i:(i+W)].copy()  # shape (n_trials, n_channels, W)
    z_w = z[:, :, i:(i+W-L+1)].copy()  # shape (n_atoms, n_trials, W-L+1)
    # pad with zeros on both sides of the last dim
    pad = L - 1
    z_pad = np.pad(z, ((0, 0), (0, 0), (pad, pad)), constant_values=0)
    # i += pad
    # shape (n_atoms, n_trials, W+L-1)
    z_w_ext = z_pad[:, :,  i:(i+W+pad)].copy()
    # shape (n_trials, n_channels, W+2*L-2)
    X_hat = construct_X_multi(z_w_ext, D=D, n_channels=n_channels)
    X_hat = X_hat[:, :, pad:pad+W]  # shape (n_trials, n_channels, W)
    residual = X_w - X_hat
    grad = _dense_transpose_convolve_z(residual, z_w)
    return grad


def test_recovery(z, D, n_part=2):
    """Compare the recovery between the full reconstructed signal X = z*D and 
    the subpart of signal reconstruction from a partition.

    Parameters
    ----------
    n_part : int
        number of partition

    Results
    -------
    None
    """
    W = int(T//n_part)
    print(f"test recovery for {n_part} partitions and a window of size {W}")

    # reconstruct full signal
    X_hat = construct_X_multi(z, D=D)
    X_hat = X_hat[:, :, :(W*n_part)]  # in case that W * n_part < T

    # pad z vector with zeros on both sides
    pad = L-1
    z_pad = np.pad(z, ((0, 0), (0, 0), (pad, pad)), constant_values=0)

    # for each partition, reconstruc the signal and compare it to original
    residuals = []
    for i in [i*W for i in range(n_part)]:
        z_w_ext = z_pad[:, :, i:(i + W + pad)].copy()
        X_part = construct_X_multi(z_w_ext, D=D, n_channels=n_channels)[
            :, :, pad:pad+W]
        res_part = X_hat[:, :, i:i+W] - X_part
        # print(f"norm of partition {int(i/W)} is {norm(res_part)}")
        residuals.append(res_part)

    residuals = np.array(residuals)

    assert norm(residuals) <= 1e-15, f"norm is {norm(residuals)}"


X, D, z = simulate_data(
    n_trials=1, n_channels=1, n_times=T, n_times_atom=L, n_atoms=1,
    n_acti_atom=n_acti_atom, random_state=42, constant_amplitude=False,
    window=True, shapes=['sin', 'gaussian'], sigma_noise=1, plot_atoms=False)

n_trials, n_channels, n_times = X.shape

for n_part in [1, 2, 4, 5, 10]:
    test_recovery(z, D, n_part)


n_part = 10
W = int(T/n_part)
X = X[:, :, (W*n_part)]  # in case that W * n_part < T
# compute full grad
full_grad = compute_grad(X, z, D, i=None)
# compute list of windowed grads
win_grads = np.array([compute_grad(X, z, D, i=this_i*W, W=W)
                      for this_i in range(T//W)])
print(
    f'norm of diff from part: {norm(win_grads.mean(axis=0)/W - full_grad/T)}')

# random
list_i_rnd = np.random.choice((T-W-L), int(T//W), replace=True) + L
win_grads = np.array([compute_grad(X, z, D, i=this_i, W=W)
                      for this_i in list_i_rnd])
print(
    f'norm of diff from random: {norm(win_grads.mean(axis=0)/W - full_grad/T)}')

win_grad = compute_grad(X, z, D, i=5000, W=W)
print(f'norm of diff: {norm(win_grad/W - full_grad/T)}')


# test reconstruction

# %%
T = 10_000
W = 1_000
list_W = [1_000, 2_000, 5_000, T]
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
