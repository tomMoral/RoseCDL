# %%
import numpy as np
import pandas as pd
from pathlib import Path

from utils_apnea import plot_multi_subject_temporal_atoms

subject_id_list = pd.read_csv(
    Path("apnea-ecg/participants.tsv"), sep='\t')['Record'].values

# 'a' (apnea), 'b' (borderline apnea), 'c' (control), 'x' (test)
for group_id in ['c']:
    subject_id_group = [id for id in subject_id_list if id[0] == group_id]
    fit_on = 'N'

    dict_d_hat = {}
    for subject_id in subject_id_group:
        subject_dir = Path(f'apnea-ecg/{subject_id}')
        subject_d_hat = np.load(subject_dir / 'd_hat.npy')
        dict_d_hat[subject_id] = subject_d_hat

    plot_multi_subject_temporal_atoms(
        dict_d_hat, save_fig=f'./group_{group_id}_{fit_on}_atoms.pdf')
# %%
