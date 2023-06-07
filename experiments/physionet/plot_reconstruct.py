# %%

import numpy as np
import json
import pickle
from pathlib import Path

from alphacsc.utils.convolution import construct_X_multi
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils.dictionary import get_lambda_max

from utils_apnea import load_ecg, plot_subject_record

# exp_id = "1f7ef48b9a0b6d977d6c48df5d610164"
exp_id = "df2b41bf54803d98c4d8c3cca654a1e8"
subject_id = "b05"

this_exp_path = Path("./benchmark_apnea-ecg") / exp_id
this_subject_dir = this_exp_path / subject_id

with open(this_exp_path / "exp_params.json", "r") as f:
    exp_params = json.load(f)

fit = exp_params["fit"]

with open(this_subject_dir / f"dict_res_{fit}.pkl", "rb") as f:
    dict_res = pickle.load(f)

print(exp_params)

#
D_hat = np.load((this_subject_dir / f"d_hat_{fit}.npy"))
D_chunk = np.load((this_subject_dir / f"d_chunk_{fit}.npy"))
z_hat = np.load((this_subject_dir / f"z_hat_{fit}.npy"))
# get power of 10 of the max
max_pow = int(np.floor(np.log10(abs(z_hat.max()))))
threshold = 10 ** (max_pow - 4)  # restrict to 5 orders of magnitude
z_hat[z_hat < threshold] = 0
X_hat = construct_X_multi(z_hat, D_hat, n_channels=1)
idx = dict_res["idx"]
if idx == "all":
    idx = list(range(X_hat.shape[0]))
plot_subject_record(subject_id, fit, idx, X_hat=X_hat)
# Compute lambda max
X, labels = load_ecg(
    subject_id,
    split=True,
    T=60,
    apply_window=True,
    verbose=False,
)
X = X[labels == fit]

lambda_max = get_lambda_max(X, D_hat, q=1)
lambda_max = np.quantile(lambda_max, q=1)
print("lambda_max:", lambda_max)

lambda_max_chunk = get_lambda_max(X, D_chunk, q=1)
lambda_max_chunk = np.quantile(lambda_max_chunk, q=1)
print("lambda_max_chunk:", lambda_max_chunk)
# %% recompute sparse code with learned dict
z_hat_rec, _, _ = update_z_multi(
    X[:20],
    D_hat,
    reg=exp_params["reg"],
    solver=exp_params["solver_z"],
    solver_kwargs=exp_params["solver_z_kwargs"],
)
X_hat_rec = construct_X_multi(z_hat_rec, D_hat, n_channels=1)

# %%
plot_subject_record(subject_id, fit, X_hat=X_hat_rec, start_trial=0, stop_trial=20)


# %%
