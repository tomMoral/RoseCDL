"""
For a subject, get th residual of its own signal to be able to better see what
the dictionary captured, on the signal used for train and on a new signal from
another session
- all signal : X.plot() et X_hat.plot
- epochs : create mne.io.RawArray(data, info)
"""
# %%

import numpy as np
from pathlib import Path
import mne

from alphacsc.init_dict import init_dictionary
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils import construct_X_multi

from wincdl.wincdl import WinCDL
from experiments.scripts.utils import get_camcan_info, get_subject_z_and_cost
from experiments.scripts.camcan import load_data as load_data_camcan

BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

# subject_id = 'sub-CC120008'
# subject_path = DATA_PATH / 'cat1' / subject_id


# %%
N_ATOMS = 40
N_TIMES_ATOM = 150
lmbd = 0.1
n_channels = 204

N_JOBS = 40
DEVICE = "cuda:1"

subject_path = SUBJECTS_PATH[0]
subject_id = subject_path.name.split('.')[0]
subject_dir = Path(f'../alphacsc_results/{subject_id}')

X = np.load(subject_path)
X /= X.std()
if X.ndim == 2:
    (n_channels, n_times) = X.shape
    n_trials = 1
elif X.ndim == 3:
    (n_trials, n_channels, n_times) = X.shape

D_init_path = subject_dir / ('D_init.npy')
try:
    D_init = np.load(D_init_path)
except:
    D_init = init_dictionary(
        X[None, :], N_ATOMS, N_TIMES_ATOM, uv_constraint='separate',
        rank1=True, window=True, D_init='chunk', random_state=None)

# %%

CDL = WinCDL(
    n_components=N_ATOMS,
    kernel_size=N_TIMES_ATOM,
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

losses, list_D, times = CDL.fit(X)

# get "true" raw object
# info, raw = get_camcan_info(subject_id, return_raw=True)
# event_id = {
#     'audiovis/1200Hz': 1,  # bimodal
#     'audiovis/300Hz': 2,   # bimodal
#     'audiovis/600Hz': 3,   # bimodal
#     'button': 4,           # button press
#     'catch/0': 5,          # unimodal auditory
#     'catch/1': 6           # unimodal visual
# }
# events, _ = mne.events_from_annotations(raw)
# events = mne.pick_events(events, include=list(event_id.values()))
# paths to CamCAN files for Inria Saclay users
DATA_DIR = Path("/storage/store/data/")
SSS_CAL = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
BIDS_root = DATA_DIR / "camcan/BIDSsep/smt/"
PARTICIPANTS_FILE = BIDS_root / "participants.tsv"
load_params = dict(sfreq=150.)
raw, info = load_data_camcan(
    BIDS_root, SSS_CAL, CT_SPARSE, subject_id, return_array=False,
    **load_params)
# info['temp']['event_id'].update(
#     {'audio': (1, 2, 3, 5), 'vis': (1, 2, 3, 6)})
epochs = mne.Epochs(
    raw, events=info['temp']['events'], event_id=info['temp']['event_id'], tmin=-0.2, tmax=0.5, preload=True)
audiovis_evoked = epochs['audiovis/1200Hz'].average()
audiovis_evoked.plot_joint(picks='meg')
# reconstruct raw
X_raw = mne.io.RawArray(X, info)

z_hat, _, _ = update_z_multi(
    X[None, :],
    CDL.D_hat_.astype(np.float64), reg=lmbd,
    solver='lgcd', solver_kwargs={'tol': 1e-3, 'max_iter': 10_000},
    n_jobs=1
)
X_hat = construct_X_multi(z_hat, D=CDL.D_hat_, n_channels=n_channels)
X_hat_raw = mne.io.RawArray(X_hat, info)

# %%
