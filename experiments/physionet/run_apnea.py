# %%
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm

from alphacsc.learn_d_z import compute_X_and_objective
from alphacsc.init_dict import init_dictionary
from dripp.cdl.utils_plot import plot_z_boxplot

from utils_apnea import load_ecg, get_subject_info, run_cdl

# download apnea-ecg from physionet
# wfdb.io.dl_database(db_dir='apnea-ecg', dl_dir='./apnea-ecg')

subject_id_list = pd.read_csv(
    Path("apnea-ecg/participants.tsv"), sep='\t')['Record'].values

# 'a' (apnea), 'b' (borderline apnea), 'c' (control), 'x' (test)
group_id = 'a'
if group_id is not None:
    subject_id_list = [id for id in subject_id_list if id[0] == group_id]

cdl_params = dict(
    n_atoms=3,
    n_times_atom=75,  # 1 s. at 100 Hz
    reg=0.1,
    n_iter=200,
    n_jobs=5,
    random_state=42,
    ds_init='chunk',
    verbose=0
)

def proc(subject_id):
    subject_dir = Path(f'apnea-ecg/{subject_id}')
    subject_dir.mkdir(parents=True, exist_ok=True)
    if (subject_dir / ('D_init.npy')).exists():
        return None

    # load data
    X, labels = load_ecg(subject_id, verbose=False)

    # fit CDL model and save results
    pobj, times, d_hat, z_hat, reg = run_cdl(
        X, cdl_params, labels=labels, fit_on='N', save_fig=subject_dir)
    plot_z_boxplot(z_hat.swapaxes(0, 1), type='box', add_points=False,
                   fig_name=subject_dir / 'z_boxplot.pdf')

    np.save(subject_dir / 'd_hat', d_hat)
    dict_res = dict(cost_init=pobj[0],
                    cost=pobj[-1],
                    compute_time=np.cumsum(times)[-1],
                    reg=reg)
    with open(subject_dir / 'dict_res.pkl', 'wb') as fp:
        pickle.dump(dict_res, fp)

    return dict_res

data = []
for subject_id in tqdm(subject_id_list):
    dict_res = proc(subject_id)
    data.append(dict_res)

    df_cost = pd.DataFrame(data=data)
    if group_id is not None:
        name = f'df_cost_self_{group_id}.csv'
    else:
        name = 'df_cost_self.csv'
    df_cost.to_csv(name, index=False)
# %%
