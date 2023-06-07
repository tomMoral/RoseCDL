# %%
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from joblib import Parallel, delayed, hash
from tqdm import tqdm
import argparse
from itertools import product

from alphacsc.learn_d_z import compute_X_and_objective
from alphacsc.init_dict import init_dictionary
from alphacsc.utils.signal import split_signal
from dripp.cdl.utils_plot import plot_z_boxplot

from utils_apnea import load_ecg, get_subject_info, run_cdl
from utils_apnea import plot_loss_history, plot_temporal_atoms

# %%
# download apnea-ecg from physionet
# wfdb.io.dl_database(db_dir='apnea-ecg', dl_dir='./apnea-ecg')

# parser = argparse.ArgumentParser()
# parser.add_argument("--subject", type=str, default=None)
# parser.add_argument(
#     "--group",
#     type=str,
#     choices=["a", "b", "c", "x"],
#     help="group id to run the CDL on ('a': apnea, 'b': borderline apnea, 'c': control, 'x': test)",
#     default=None,
# )
# parser.add_argument("--n_subjects", type=int, default=None)
# parser.add_argument("--n_atoms", type=int, default=10)
# parser.add_argument("--n_times_atom", type=int, default=75)
# parser.add_argument("--n_iter", type=int, default=100)
# parser.add_argument("--reg", type=float, default=0.1)
# parser.add_argument(
#     "--fit",
#     type=str,
#     default="N",
#     choices=["N", "A", "all"],
#     help="'A': apnea minutes, 'N': non-apnea minutes, 'all': both apnea and non-apnea minutes",
# )

# args = parser.parse_args()
# subject_id = args.subject
# group_id = args.group
# n_subjects = args.n_subjects
# fit = args.fit
# n_iter = args.n_iter
# reg = args.reg


group_id = "b"
subject_id = None
n_subjects = None
n_atoms = 10
n_times_atom = 75

MAX_TRIALS = 20

if subject_id is not None:
    subject_id_list = np.atleast_1d(subject_id)
else:
    subject_id_list = pd.read_csv(Path("apnea-ecg/participants.tsv"), sep="\t")[
        "Record"
    ].values
    group_des = dict(a="apnea", b="borderline apnea", c="control", x="test")

    if group_id is not None:
        subject_id_list = [id for id in subject_id_list if id[0] == group_id]
        print(f"Run CDL on group {group_id} ({group_des[group_id]})")
    else:
        print("Run CDL on all subjects")

if n_subjects is not None:
    subject_id_list = subject_id_list[:n_subjects]


# for n_iter, fit, reg in product([200, 500], ['A', 'N'], [0.1, 1]):
# for n_iter, fit, reg in product([200, 500], ["A", "N"], [10, 25, 50]):
# for n_iter, fit, reg in product([1_000], ["A", "N"], [25]):
for n_iter, fit, reg in product([1_000], ["A", "N"], [5]):
    print(n_iter, fit, reg)
    cdl_params = dict(
        # Problem Specs
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        rank1=False,
        window=True,  # if True, apply a temporal window reparametrization
        uv_constraint="auto",
        sort_atoms=True,  # if True, sort atoms by explained variances
        # Global algorithm
        algorithm="batch",
        n_iter=100,  # number of iteration for the alternate minimization
        eps=1e-4,  # cvg threshold
        reg=reg,
        lmbd_max="fixed",
        # Z-step parameters
        solver_z="fista",
        solver_z_kwargs={"tol": 1e-3, "max_iter": n_iter},  # tol et max_iter
        unbiased_z_hat=True,
        # D-step parameters
        solver_d="fista",
        solver_d_kwargs={"eps": 1e-3, "max_iter": 100, "resample_strategy": "chunk"},
        D_init="chunk",
        # Technical parameters
        n_jobs=5,
        verbose=False,
        random_state=None,
    )

    exp_params = dict(group_id=group_id, fit=fit, **cdl_params)
    exp_dir = Path(f"benchmark_apnea-ecg/{hash(exp_params)}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(exp_dir)

    # save experiment parameters
    with open(exp_dir / "exp_params.json", "w") as outfile:
        json.dump(exp_params, outfile, indent=4)

    def proc(subject_id, cdl_params):
        subject_dir = exp_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        if (subject_dir / ("d_init.npy")).exists():
            return None

        # load data
        if fit in ["A", "N"]:
            X, labels = load_ecg(
                subject_id,
                split=True,
                T=60,
                apply_window=True,
                verbose=False,
            )
            X = X[labels == fit]
        elif fit == "all":
            X = load_ecg(subject_id, split=False, apply_window=True, verbose=False)
            X = split_signal(X, n_splits=10, apply_window=True)
            labels = None

        if MAX_TRIALS < X.shape[0]:
            idx = np.random.choice(list(range(X.shape[0])), MAX_TRIALS, replace=False)
            X = X[idx]
        else:
            idx = "all"

        X /= X.std()  # shape (n_splits, n_channels, n_times)

        # compute random dict and save
        d_random = init_dictionary(
            X,
            cdl_params["n_atoms"],
            cdl_params["n_times_atom"],
            D_init="random",
            rank1=False,
            random_state=cdl_params["random_state"],
        )
        np.save(subject_dir / f"d_random_{fit}", d_random)

        # compute chunk dict and save
        d_chunk = init_dictionary(
            X,
            cdl_params["n_atoms"],
            cdl_params["n_times_atom"],
            D_init="chunk",
            rank1=False,
            random_state=cdl_params["random_state"],
        )
        np.save(subject_dir / f"d_chunk_{fit}", d_chunk)

        # set init dictionary
        if cdl_params["D_init"] == "random":
            cdl_params["D_init"] = d_random
        elif cdl_params["D_init"] == "chunk":
            cdl_params["D_init"] = d_chunk

        np.save(subject_dir / f"d_init_{fit}", cdl_params["D_init"])

        # fit CDL model and save results
        pobj, times, d_hat, z_hat, reg = run_cdl(X, cdl_params)
        np.save(subject_dir / f"pobj_{fit}", pobj)
        np.save(subject_dir / f"times_{fit}", times)
        np.save(subject_dir / f"z_hat_{fit}", z_hat)
        np.save(subject_dir / f"d_hat_{fit}", d_hat)
        # save global res dictionary
        dict_res = dict(
            subject_id=subject_id,
            cost_init=pobj[0],
            cost=pobj[-1],
            compute_time=np.cumsum(times)[-1],
            reg=reg,
            idx=idx,
        )
        with open(subject_dir / f"dict_res_{fit}.pkl", "wb") as fp:
            pickle.dump(dict_res, fp)

        # plot loss history
        # plot_loss_history(
        #     [pobj], times=[times], save_fig=subject_dir / f"loss_history_{fit}.pdf"
        # )
        # plot learned atoms
        # plot_temporal_atoms(d_hat, save_fig=subject_dir / f"atoms_{fit}.pdf")
        # plot z activation values boxplots
        # plot_z_boxplot(
        #     z_hat.swapaxes(0, 1),
        #     type="box",
        #     add_number=True,
        #     fig_name=subject_dir / f"z_boxplot_{fit}.pdf",
        # )
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
        name += f"_{fit}_{n_iter}_{reg}.csv"
        df_cost.to_csv(name, index=False)
# %%
