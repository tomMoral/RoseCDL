# %%
import numpy as np
from pathlib import Path

from alphacsc.utils.dictionary import get_uv
from alphacsc.utils.convolution import sort_atoms_by_explained_variances
from dripp.cdl.plotting import display_atoms

from experiments.scripts.utils import \
    get_lambda_global, get_D_sub, get_subject_z_and_cost, get_camcan_info
from experiments.scripts.utils_plot import plot_z_boxplot


BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

N_ATOMS = 40
N_TIMES_ATOM = 150

n_subjects = 10
list_subjects_path = SUBJECTS_PATH[:n_subjects]
reg = 0.1
method = np.median
LMBD, list_lmbd = get_lambda_global(
    list_subjects_path, N_ATOMS, N_TIMES_ATOM, reg=reg, method=method)

dict_lmbd = {reg: reg * method(list_lmbd) for reg in [0.1, 0.3, 0.5, 1]}

# %%

for subject_path in list_subjects_path:

    subject_id = subject_path.name.split('.')[0]
    info = get_camcan_info(subject_id)

    for reg in [0.1, 0.3, 0.5]:
        _, D_hat_sub, lmbd = get_D_sub(
            subject_path, n_atoms=N_ATOMS, n_times_atom=N_TIMES_ATOM,
            lmbd=dict_lmbd[reg])

        subject_id, z_hat, cost = get_subject_z_and_cost(
            subject_path, D_hat_sub, reg=lmbd)

        N_CHANNELS = D_hat_sub.shape[1]
        D_hat_sub, z_hat = sort_atoms_by_explained_variances(
            D_hat_sub, z_hat, N_CHANNELS)

        plot_z_boxplot(z_hat, p_threshold=0, per_atom=True,
                       yscale='log', add_points=False, add_number=True,
                       fig_name=f"z_boxplot_{subject_id}_reg{reg}.pdf")

        uv = get_uv(D_hat_sub)
        dict_cdl = dict(u_hat_=uv[:, :N_CHANNELS],
                        v_hat_=uv[:, N_CHANNELS:])
        # plot atoms
        plotted_atoms = list(range(N_ATOMS))
        display_atoms(dict_cdl, info, plotted_atoms, sfreq=150.,
                      plot_spatial=True, plot_temporal=True, plot_psd=False,
                      save_plot=True,
                      fig_name=f"atoms_{subject_id}_reg{reg}.pdf",
                      dir_path='', dpi=300)
# %%
