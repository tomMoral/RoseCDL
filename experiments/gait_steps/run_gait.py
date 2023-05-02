"""
https://charles.doffy.net/files/ipol-walk-data-2019.pdf
"""
# %%
from wincdl.wincdl import WinCDL
from dicodile.utils.viz import display_dictionaries
from alphacsc import BatchCDL
from alphacsc.init_dict import init_dictionary
from alphacsc.loss_and_gradient import compute_X_and_objective_multi
from alphacsc.utils.convolution import sort_atoms_by_explained_variances
import matplotlib.pyplot as plt
import numpy as np

from utils_gait import get_gait_data, get_participants




participants = get_participants()
subjects_healthy = participants[participants['PathologyGroup']
                                == 'Healthy']['Subject'].values

subject = subjects_healthy[0]
trial = 1
X = get_gait_data(subject, trial, verbose=False)['data']['RAV'].to_numpy()
X /= X.std()
# reshape X to (n_trials, n_channels, n_times)
X_csc = X.reshape(1, 1, *X.shape)
X_win =  X.reshape(1, *X.shape)
n_channels, n_times = X_win.shape

# X = X.reshape(1, 1, *X.shape)
# n_trials, n_channels, n_times = X.shape
# print('(n_trials, n_channels, n_times):', X.shape)
# %%
WINDOW = True
N_ATOMS = 8
N_TIMES_ATOM = 200


D_init = init_dictionary(X_csc,
                         n_atoms=N_ATOMS,
                         n_times_atom=N_TIMES_ATOM,
                         rank1=False,
                         window=WINDOW,
                         D_init='chunk',
                         random_state=60)

# cdl_params = dict(
#     # Problem Specs
#     n_atoms=N_ATOMS,
#     n_times_atom=N_TIMES_ATOM,
#     rank1=False,
#     window=WINDOW,
#     uv_constraint='auto',
#     # Global algorithm
#     n_iter=50,
#     eps=1e-10,
#     reg=0.1,
#     lmbd_max='scaled',
#     # Z-step parameters
#     solver_z='dicodile',
#     solver_z_kwargs={'max_iter': 10_000, 'eps': 1e-7},
#     unbiased_z_hat=False,
#     # D-step parameters
#     solver_d='fista',
#     solver_d_kwargs={'max_iter': 300, 'eps': 1e-8, 'momentum': False},
#     D_init=D_init,
#     # Technical parameters
#     n_jobs=4,
#     random_state=60,
#     sort_atoms=True
# )
# dicod_cdl = BatchCDL(**cdl_params)
# n_times_valid = n_times - N_TIMES_ATOM + 1
# z_init = np.zeros(shape=(1, N_ATOMS, n_times_valid))
# cost, X_hat = compute_X_and_objective_multi(
#     X_csc, z_init, D_hat=D_init, reg=cdl_params['reg'], feasible_evaluation=True,
#     uv_constraint='auto', return_X_hat=True)

# import torch
# loss_fn = torch.nn.MSELoss(reduction='sum')
# X_csc_torch = torch.tensor(
#                 X_csc,
#                 dtype=torch.float,
#                 device='cuda:1'
#             )
# X_hat_torch = torch.tensor(
#                 X_hat,
#                 dtype=torch.float,
#                 device='cuda:1'
#             )
# cost_torch = loss_fn(X_hat_torch, X_csc_torch) / 2

# res = dicod_cdl.fit(X_csc)
# D_hat_dicod = res._D_hat

# fig = display_dictionaries(D_init, D_hat_dicod)
# plt.show()

# plt.plot(dicod_cdl.pobj_[::2])
# plt.xlabel('Iterations')
# plt.ylabel('Lasso loss')
# plt.show()
# # %%
# cdl_params.update(dict(
#     reg=dicod_cdl.reg_,
#     lmbd_max='fixed',
#     solver_z='fista',
#     solver_z_kwargs={'max_iter': 2_000, 'eps': 1e-7},
# ))
# fista_cdl = BatchCDL(**cdl_params)

# res = fista_cdl.fit(X_csc)
# D_hat_fista = res._D_hat

# fig = display_dictionaries(D_init, D_hat_fista)
# plt.show()

# plt.plot(fista_cdl.pobj_[::2])
# plt.xlabel('Iterations')
# plt.ylabel('Lasso loss')
# plt.show()
# %%

gait_cdl = WinCDL(
    n_components=N_ATOMS,
    kernel_size=N_TIMES_ATOM,
    n_channels=n_channels,
    # lmbd=dicod_cdl.reg_,
    lmbd=1.5134614391157055,
    n_iterations=1_000,  # Fista iterations for z step
    epochs=40,
    max_batch=1,
    stochastic=True,  # if false, fit on full signal
    optimizer="linesearch",
    lr=0.1,
    mini_batch_window=n_times,  # full signal
    mini_batch_size=1,
    device='cuda:1',
    rank='full',
    window=True,
    D_init=D_init,
    list_D=True,
    dimN=1,
)
losses, list_D, times = gait_cdl.fit(X_win)


# sort_atoms = cdl_params['sort_atoms']
sort_atoms = False
if sort_atoms:
    D_hat_torch, z_hat_torch = sort_atoms_by_explained_variances(
                list_D[-1], gait_cdl.csc.z.cpu().numpy(), n_channels=n_channels)
else:
    D_hat_torch = list_D[-1]
    z_hat_torch = gait_cdl.csc.z.cpu().numpy()

fig = display_dictionaries(D_init, D_hat_torch)
# fig = display_dictionaries(D_init, D_hat_dicod, D_hat_fista, D_hat_torch)
plt.show()

plt.plot(losses)
# plt.yscale('log')
plt.ylabel('Lasso loss')
plt.xlabel('Epochs')
plt.show()
# %%

plt.plot(-np.ones(1))
plt.plot(dicod_cdl.pobj_[::2], label='dicodile')
plt.plot(fista_cdl.pobj_[::2], label='fista')
plt.plot(losses, label='torch')
plt.yscale('log')
# plt.xlim(0, None)
plt.xlabel('Iterations')
plt.ylabel('Lasso loss')
plt.legend()
plt.show()

# %%
xx = np.array(range(1, N_ATOMS+1))
plt.plot(xx, -np.ones(xx.shape))
# plt.plot(xx, np.count_nonzero(dicod_cdl.z_hat_, axis=2)[0], label='dicodile')
# plt.plot(xx, np.count_nonzero(fista_cdl.z_hat_, axis=2)[0], label='fista')
plt.plot(xx, np.count_nonzero(z_hat_torch, axis=2)[0], label='torch')
plt.xlabel('Atom id')
plt.ylabel('# non-zero activations')
plt.xlim(1, N_ATOMS)
plt.ylim(0, None)
plt.legend()
plt.show()

# %%
