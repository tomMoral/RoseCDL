"""
Compute the evolution of cost as the number of atoms learned
"""
# %%
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path

from alphacsc.utils.convolution import sort_atoms_by_explained_variances
from dripp.cdl.plotting import display_atoms
# from dripp.cdl.camcan import load_data as load_data_camcan

from experiments.scripts.utils import \
    get_lambda_global, get_D_sub, get_subject_z_and_cost

BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

# paths to CamCAN files for Inria Saclay users
DATA_DIR = Path("/storage/store/data/")
SSS_CAL = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
BIDS_root = DATA_DIR / "camcan/BIDSsep/smt/"


N_ATOMS = 40
N_TIMES_ATOM = 150

N_JOBS = 40
DEVICE = "cuda:1"


n_subjects = 10
list_subjects_path = SUBJECTS_PATH[:n_subjects]
list_n_atoms = [40, 70, 100]

X = np.load(SUBJECTS_PATH[0])
N_CHANNELS = X.shape[0]

LMBD = get_lambda_global(
    list_subjects_path, N_ATOMS, N_TIMES_ATOM, method=np.median)
print(f"Global lambda over {len(list_subjects_path)} subjects is {LMBD}")
# %%
res = []
for subject_path in list_subjects_path:
    # get info
    # load_params = dict(sfreq=150.)
    # _, info = load_data_camcan(
    #     BIDS_root, SSS_CAL, CT_SPARSE, subject_id, **load_params)

    for this_n_atoms in list_n_atoms:
        D_init, D_hat_sub, _ = get_D_sub(
            subject_path, n_atoms=this_n_atoms, n_times_atom=N_TIMES_ATOM,
            lmbd=LMBD)
        subject_id, z_hat, cost = get_subject_z_and_cost(
            subject_path, D_hat_sub, reg=LMBD)
        # D_hat_sub, z_hat = sort_atoms_by_explained_variances(
        #     D_hat_sub, z_hat, N_CHANNELS)
        # # plot atoms
        # plotted_atoms = list(range(this_n_atoms))
        # display_atoms(cdl, info, plotted_atoms, s_freq,
        #               plot_spatial=True, plot_temporal=True, plot_psd=False,
        #               save_plot=True,
        #               fig_name=f"sample_atoms_{torch_cdl_params['epochs']}_{reg}.pdf",
        #               dir_path='', dpi=300)
        res.append(dict(
            subject_id=subject_id,
            cost=cost,
            n_atoms=this_n_atoms,
        ))
# %%
df_res = pd.DataFrame(res)
df_res.to_csv('df_res_n_atoms.csv')
