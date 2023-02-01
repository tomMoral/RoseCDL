# %%
from alphacsc.datasets.mne_data import load_data as load_data_mne

from wincdl.wincdl import WinCDL

load_params = dict(sfreq=150., n_splits=1)
X, info = load_data_mne(dataset='sample', **load_params)

(n_trials, n_channels, n_times) = X.shape

# %%
n_atoms = 40
n_times_atom = 150
lmbd = 0.1

CDL = WinCDL(
    n_components=n_atoms,
    kernel_size=n_times_atom,
    n_channels=n_channels,
    lmbd=lmbd,
    n_iterations=20,
    epochs=100,
    max_batch=10,
    stochastic=False,
    optimizer="linesearch",
    lr=0.1,
    gamma=0.9,
    mini_batch_window=1_000,
    mini_batch_size=1,
    device=DEVICE,
    rank="uv_constraint",
    window=True,
    D_init=D_init,
    positive_z=True,
    list_D=False,
    dimN=1
)
