import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from alphacsc.update_z_multi import update_z_multi

from dripp.dripp_model import DriPP
from dripp.config import BIDS_root
from dripp.cdl.camcan import get_events
from dripp.cdl.utils import post_process_cdl


BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

N_ATOMS = 40
N_TIMES_ATOM = 150

n_subjects = 20
list_subjects_path = SUBJECTS_PATH[:n_subjects]

# %% test

subject_path = list_subjects_path[0]
subject_id = subject_path.name.split('.')[0]
subject_dir = Path(f'./{subject_id}')
reg, lmbd_max = 0.1, 'scaled'
suff = f"reg_{reg}_lmbd_{lmbd_max}"

uv_hat_ = np.load(subject_dir / ('uv_hat_' + suff + '.npy'))
X = np.load(subject_path)
X /= X.std()
n_channels = X.shape[X.ndim - 2]
try:
    z_hat = np.load(subject_dir / ('z_hat' + suff + '.npy'))
except:
    z_hat, _, _ = update_z_multi(
        X[None, :],
        uv_hat_.astype(np.float64), reg=reg,
        solver='lgcd', solver_kwargs={'tol': 1e-4, 'max_iter': 1_000_000},
        n_jobs=1
    )
    np.save(subject_dir / ('z_hat' + suff), z_hat)
# %%

u_hat_, v_hat_ = uv_hat_[:, :n_channels], uv_hat_[:, n_channels:]


# get subject's event dict
sfreq = 150.
events, event_id = get_events(BIDS_root, subject_id, sfreq=sfreq)

# pre-process
post_process_params = dict(
    time_interval=0.01, threshold=0, percent=True, per_atom=True)
events_tt, atoms_tt = post_process_cdl(
    events=events,
    event_id=event_id.values(),
    v_hat_=v_hat_, z_hat=z_hat,
    sfreq=sfreq,
    post_process_params=post_process_params
)

dripp_ = DriPP()
dripp_.fit(acti_tt, driver_tt, T=)
