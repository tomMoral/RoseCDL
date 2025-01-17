# %%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from alphacsc import GreedyCDL
from alphacsc.utils.signal import split_signal


from dripp.cdl.plotting import display_atoms

from experiments.scripts.utils_plot import plot_z_boxplot
from experiments.scripts.utils import get_camcan_info

BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

N_ATOMS = 40
N_TIMES_ATOM = 150

n_subjects = 20
list_subjects_path = SUBJECTS_PATH[:n_subjects]

LMBD = 14.78
# %%
cdl_params = {
    # Shape of the dictionary
    'n_atoms': N_ATOMS,
    'n_times_atom': N_TIMES_ATOM,
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    'rank1': True,
    'uv_constraint': 'separate',
    # apply a temporal window reparametrization
    'window': True,
    # at the end, refit the activations with fixed support
    # and no reg to unbias
    'unbiased_z_hat': True,
    # Initialize the dictionary with random chunk from the data
    'D_init': 'chunk',
    # rescale the regularization parameter to be a percentage of lambda_max
    'lmbd_max': "scaled",  # original value: "scaled"
    'reg': 0.1,
    # Number of iteration for the alternate minimization and cvg threshold
    'n_iter': 100,  # original value: 100
    'eps': 1e-4,  # original value: 1e-4
    # solver for the z-step
    'solver_z': "lgcd",
    'solver_z_kwargs': {'tol': 1e-3,  # stopping criteria
                        'max_iter': 100000},
    # solver for the d-step
    'solver_d': 'alternate_adaptive',
    'solver_d_kwargs': {'max_iter': 300},  # original value: 300
    # sort atoms by explained variances
    'sort_atoms': True,
    # Technical parameters
    'verbose': 1,
    'random_state': 0,
    'n_jobs': 10
}


subject_id = 'sub-CC120120'
subject_path = DATA_PATH / 'cat1' / f'{subject_id}.npy'
info = get_camcan_info(subject_id)

X = np.load(subject_path)
X -= X.mean()
X /= X.std()
X_split = split_signal(X, n_splits=10, apply_window=True)

subject_dir = Path(f'./{subject_id}')
D_init = np.load(subject_dir / 'D_init.npy')

lmbd_max = 'fixed'
reg = 0.3
suff = f"reg_{reg}_lmbd_{lmbd_max}"
if lmbd_max == 'fixed':
    suff += f'_{str(LMBD)}'
reg *= LMBD

cdl_params.update(reg=reg, lmbd_max=lmbd_max, D_init=D_init)

cdl = GreedyCDL(**cdl_params)
cdl.fit(X_split)
np.save(subject_dir / ('uv_hat_' + suff), cdl.uv_hat_)

if cdl.uv_hat_.shape[0] < N_ATOMS:
    print(f"{cdl.uv_hat_.shape[0]} atoms were learnt for {suff}")

# compute atoms activation intensities
z_hat = cdl.transform(X[None, :])
plot_z_boxplot(
    z_hat, p_threshold=0, per_atom=True, yscale='log',
    add_points=False, add_number=True,
    fig_name=subject_dir / f"z_boxplot_{subject_id}_{suff}.pdf")

plotted_atoms = list(range(N_ATOMS))
display_atoms(cdl, info, plotted_atoms, sfreq=150.,
              plot_spatial=True, plot_temporal=True,
              plot_psd=False, save_plot=True,
              fig_name=f"atoms_{subject_id}_{suff}.pdf",
              dir_path=subject_dir, dpi=300)

# plot loss curves
pobj, times = cdl._pobj, cdl._times
np.save(subject_dir / ('pobj_' + suff), pobj)
np.save(subject_dir / ('times_' + suff), times)
try:
    idx_stop = np.where(np.diff(pobj) >= -1e-5)[0][0]
    suff_titre = f'(up to the {idx_stop}-th iteration)'
except IndexError:
    idx_stop = -1
    suff_titre = ''
plt.plot(np.cumsum(times)[:idx_stop], pobj[:idx_stop])
plt.xlabel('Times')
plt.xlim(0, None)
plt.title(f'Objective function {suff_titre}')
plt.savefig(subject_dir / (f'loss_curve_{suff}.pdf'))
plt.show()
# %%
