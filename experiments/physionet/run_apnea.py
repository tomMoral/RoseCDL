# %%
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import argparse

from alphacsc.init_dict import init_dictionary
from dripp.cdl.utils_plot import plot_z_boxplot

from utils_apnea import load_ecg, run_cdl
# %%
# download apnea-ecg from physionet
# wfdb.io.dl_database(db_dir='apnea-ecg', dl_dir='./apnea-ecg')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--group",
    type=str,
    choices=["a", "b", "c", "x"],
    help="group id to run the CDL on ('a': apnea, 'b': borderline apnea, 'c': control, 'x': test)",
)
parser.add_argument("--n_atoms", type=int, default=3)
parser.add_argument("--n_times_atom", type=int, default=75)
parser.add_argument("--n_iter", type=int, default=200)
parser.add_argument(
    "--fit",
    type=str,
    default="N",
    choices=["N", "A"],
    help="'A': apnea minutes, 'N': non-apnea minutes",
)

args = parser.parse_args()
group_id = args.group

subject_id_list = pd.read_csv(Path("apnea-ecg/participants.tsv"), sep="\t")[
    "Record"
].values
group_des = dict(a="apnea", b="borderline apnea", c="control", x="test")

if group_id is not None:
    subject_id_list = [id for id in subject_id_list if id[0] == group_id]
    print(f"Run CDL on group {group_id} ({group_des[group_id]})")
else:
    print("Run CDL on all subjects")

cdl_params = dict(
    n_atoms=args.n_atoms,
    n_times_atom=args.n_times_atom,
    reg=0.1,
    n_iter=args.n_iter,
    n_jobs=5,
    random_state=42,
    ds_init="chunk",
    verbose=0,
)


def proc(subject_id, cdl_params):
    subject_dir = Path(f"apnea-ecg/{subject_id}")
    subject_dir.mkdir(parents=True, exist_ok=True)
    if (subject_dir / ("D_init.npy")).exists():
        return None

    # load data
    X, labels = load_ecg(subject_id, verbose=False)
    # X /= X.std()

    # compute random dict and save
    X_ = X.squeeze()[labels == args.fit]
    X_ /= X_.std()
    d_random = init_dictionary(
        X_[:, None, :],
        cdl_params["n_atoms"],
        cdl_params["n_times_atom"],
        D_init="random",
        rank1=False,
    )
    np.save(subject_dir / f"d_random_{args.fit}", d_random[:, 0, :])

    # fit CDL model and save results
    pobj, times, d_hat, z_hat, reg = run_cdl(
        X, cdl_params, labels=labels, fit_on=args.fit, save_fig=subject_dir
    )
    plot_z_boxplot(
        z_hat.swapaxes(0, 1),
        type="box",
        add_points=False,
        fig_name=subject_dir / "z_boxplot.pdf",
    )

    np.save(subject_dir / f"d_hat_{args.fit}", d_hat)
    dict_res = dict(
        subject_id=subject_id,
        cost_init=pobj[0],
        cost=pobj[-1],
        compute_time=np.cumsum(times)[-1],
        reg=reg,
    )
    with open(subject_dir / f"dict_res_{args.fit}.pkl", "wb") as fp:
        pickle.dump(dict_res, fp)

    return dict_res


data = []
for subject_id in tqdm(subject_id_list):
    dict_res = proc(subject_id, cdl_params)
    data.append(dict_res)

    df_cost = pd.DataFrame(data=data)
    name = "df_cost_self"
    if group_id is not None:
        name += f"_{group_id}"
    else:
        name += "_all"
    name += f"_{args.fit}.csv"
    df_cost.to_csv(name, index=False)
# %%
