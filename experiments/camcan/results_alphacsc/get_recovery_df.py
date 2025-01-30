# %%
from pathlib import Path

import numpy as np
import pandas as pd
from alphacsc.init_dict import init_dictionary
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.scripts.utils import get_subject_z_and_cost

BASE_PATH = Path("/storage/store2/work/bmalezie")
DICT_PATH = BASE_PATH / "cdl-population/results/camcan"
DATA_PATH = BASE_PATH / "camcan-cdl"
SUBJECTS_PATH = [x for x in DATA_PATH.glob("**/*") if x.is_file()]

N_ATOMS = 40
N_TIMES_ATOM = 150

n_subjects = 20
list_subjects_path = SUBJECTS_PATH[:n_subjects]

# LMBD, list_lmbd = get_lambda_global(
#     list_subjects_path, N_ATOMS, N_TIMES_ATOM, reg=1, method=np.median)
# print(LMBD)
LMBD = 144.48
# %%


def my_func(subject_path, list_subjects_path, reg, lmbd_max):
    # subject_path = list_subjects_path[i]
    subject_id = subject_path.name.split(".")[0]

    subject_dir = Path(f"./{subject_id}")
    suff = f"reg_{reg}_lmbd_{lmbd_max}"

    if lmbd_max == "fixed":
        reg *= LMBD

    X = np.load(subject_path)
    X /= X.std()
    if X.ndim == 2:
        (n_channels, n_times) = X.shape
        n_trials = 1
    elif X.ndim == 3:
        (n_trials, n_channels, n_times) = X.shape

    # get initial dictionary with alphacsc
    D_random = init_dictionary(
        X[None, :],
        N_ATOMS,
        N_TIMES_ATOM,
        uv_constraint="separate",
        rank1=True,
        window=True,
        D_init="random",
        random_state=None,
    )

    D_init = np.load(subject_dir / ("D_init.npy"))
    D_hat_sub = np.load(subject_dir / ("uv_hat_" + suff + ".npy"))

    results = Parallel(n_jobs=5)(
        delayed(get_subject_z_and_cost)(subject_path, D_hat_sub, reg=reg)
        for subject_path in list_subjects_path
    )

    dic_res = []
    for res in results:
        dic_res.append(dict(subject_id=res[0], dict_fit=subject_id, cost=res[2]))

    cost_init = get_subject_z_and_cost(subject_path, D_init, reg=reg)[2]
    dic_res.append(dict(subject_id=subject_id, dict_fit="D_init", cost=cost_init))

    cost_random = get_subject_z_and_cost(subject_path, D_random, reg=reg)[2]
    dic_res.append(dict(subject_id=subject_id, dict_fit="D_random", cost=cost_random))

    return dic_res


lmbd_max = "fixed"
for reg in [0.1, 0.3]:
    suff = f"reg_{reg}_lmbd_{lmbd_max}"

    results = [
        my_func(subject_path, list_subjects_path, reg, lmbd_max)
        for subject_path in tqdm(list_subjects_path)
    ]
    data = []
    for dic_res in results:
        for this_dict in dic_res:
            data.append(this_dict)

    df_cost = pd.DataFrame(data=data)
    df_cost.to_csv(f"df_cost_{suff}.csv", index=False)
