""" 
For a set a subjects, compute the recovery cost of one's dictionary on
another's signal, and save the matrix as a dataframe.
"""
# %%
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm

from alphacsc.update_z import update_z
from alphacsc.learn_d_z import compute_X_and_objective, init_dictionary

from utils_apnea import load_ecg

subject_id_list = pd.read_csv(
    Path("apnea-ecg/participants.tsv"), sep='\t')['Record'].values

# 'a' (apnea), 'b' (borderline apnea), 'c' (control), 'x' (test)
group_id = 'x'
subject_id_group = [id for id in subject_id_list if id[0] == group_id]
fit_on = 'N'

subjects_rows = []
for subject_id in subject_id_group:
    subject_dir = Path(f'apnea-ecg/{subject_id}')
    with open(subject_dir / 'dict_res.pkl', 'rb') as f:
        dict_res = pickle.load(f)

    subjects_rows.extend([dict(
        subject_id=subject_id,
        dict_fit='D_init',
        cost=dict_res['cost_init']
    ), dict(
        subject_id=subject_id,
        dict_fit=subject_id,
        cost=dict_res['cost']
    )])

    # load final dictionary
    subject_d_hat = np.load(subject_dir / 'd_hat.npy')
    n_atoms, n_times_atom = subject_d_hat.shape

    # get random dictionary
    X, labels = load_ecg(subject_id, verbose=False)
    X_ = X.squeeze()[labels == fit_on]
    d_random = init_dictionary(
        X_[:, None, :], n_atoms, n_times_atom, D_init='random', rank1=False)
    d_random = d_random[:, 0, :]
    z_hat = update_z(X_, d_random, reg=0.1, solver='l-bfgs',
                     solver_kwargs={'tol': 1e-4, 'max_iter': 10_000})
    cost = compute_X_and_objective(X_, z_hat, d_random, reg=0.1)
    subjects_rows.append(dict(
        subject_id=subject_id,
        dict_fit='D_random',
        cost=cost
    ))

    for other_subject_id in tqdm(subject_id_group):
        if subject_id == other_subject_id:
            continue
        X, labels = load_ecg(other_subject_id, verbose=False)
        X_ = X.squeeze()[labels == fit_on]
        z_hat = update_z(X_, subject_d_hat, reg=0.1, solver='l-bfgs',
                         solver_kwargs={'tol': 1e-4, 'max_iter': 10_000})
        cost = compute_X_and_objective(X_, z_hat, subject_d_hat, reg=0.1)
        subjects_rows.append(dict(
            subject_id=subject_id,
            dict_fit=other_subject_id,
            cost=cost
        ))

recovery_df = pd.DataFrame(data=subjects_rows)
recovery_df.to_csv(f'recovery_df_{group_id}.csv', index=False)
# %%
