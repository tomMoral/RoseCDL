# %%
from pathlib import Path
import wfdb
from alphacsc import learn_d_z

from utils_apnea import load_ecg

# download apnea-ecg from physionet
# wfdb.io.dl_database(db_dir='apnea-ecg', dl_dir='./apnea-ecg')
# %%
X, labels = load_ecg(subject="a01", T=60, data_path=Path("apnea-ecg"))

# %%
n_splits, n_channels, n_times = X.shape
if n_channels == 1:
    X = X.squeeze()

X_a, X_n = X[labels == 'A'], X[labels == 'N']


cdl_params = dict(
    n_atoms=3,
    n_times_atom=100,  # 1 s.
)
if n_channels == 1:
    pobj, times, D_hat, z_hat, reg = learn_d_z(
        X_a, n_atoms, n_times_atom, reg=0.1, n_iter=200, n_jobs=5, random_state=42)
# %%
