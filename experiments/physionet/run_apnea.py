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
from alphacsc.update_z_multi import update_z_multi
from alphacsc.loss_and_gradient import compute_X_and_objective_multi
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

n_iter_fista = 1_000
fit = "all"
reg = 10

n_trials_train = 60
n_trials_test = 20
MAX_TRIALS = 20
RANDOM_TRIAL = False

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

cdl_params = dict(
    # Problem Specs
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    rank1=False,
    window=True,  # if True, apply a temporal window reparametrization
    uv_constraint="auto",
    sort_atoms=False,  # if True, sort atoms by explained variances
    # Global algorithm
    algorithm="batch",
    n_iter=500,  # number of iteration for the alternate minimization
    eps=1e-4,  # cvg threshold
    reg=reg,
    lmbd_max="fixed",
    # Z-step parameters
    solver_z="fista",
    solver_z_kwargs={"tol": 1e-3, "max_iter": n_iter_fista},  # tol et max_iter
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

with open("apnea-ecg/bad_trials.json", "r") as f:
    bads = json.load(f)

exp_params = dict(group_id=group_id, fit=fit, **cdl_params)
# exp_dir = Path(f"benchmark_apnea-ecg/{hash(exp_params)}")
# exp_dir.mkdir(parents=True, exist_ok=True)
# print(exp_dir)

# # save experiment parameters
# with open(exp_dir / "exp_params.json", "w") as outfile:
#     json.dump(exp_params, outfile, indent=4)


def get_train_test_data(
    subject_id, n_trials_train=20, n_trials_test=0, q=None, l_freq=None, h_freq=None
):
    """

    q : float | None
        if not None, between 0 and 100 inclusive, quantile threshold for outliers
        exclusion, based on trianls' std

    l_freq, h_freq : float | None
        frequency filtering
    """
    # load data
    X, labels = load_ecg(
        subject_id,
        split=True,
        T=60,
        apply_window=True,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=False,
    )
    n_trials = X.shape[0]
    assert n_trials_train + n_trials_test <= n_trials
    valid_trials_id = list(range(n_trials))

    if q is not None:
        trials_std = X.squeeze().std(axis=1)
        q_min, q_max = np.percentile(trials_std, [q, 100 - q])
        valid_trials_id = np.where((trials_std < q_max) & (trials_std > q_min))[0]

    if RANDOM_TRIAL:
        idx = np.random.choice(
            valid_trials_id, n_trials_train + n_trials_test, replace=False
        )
    else:
        idx = valid_trials_id[: n_trials_train + n_trials_test]

    X_train = X[idx[:n_trials_train]]
    X_test = X[idx[n_trials_train:]]
    # normalize data
    X_train_std = X_train.std()
    X_train /= X_train_std
    X_test /= X_train_std

    return X_train, X_test, idx


# %%
def proc(
    subject_id,
    cdl_params,
    n_trials_train=20,
    n_trials_test=0,
    q=None,
    l_freq=None,
    h_freq=None,
    suffix="",
):
    # if fit in ["A", "N"]:
    #     subject_bads = bads.get(subject_id, [])
    #     # trial dataframe
    #     bad_col = np.zeros(len(labels), dtype=int)
    #     bad_col[subject_bads] = 1
    #     trial_df = pd.DataFrame({"labels": labels, "bad": bad_col})
    #     trial_df.index.name = "trial_id"
    #     trial_df.reset_index(inplace=True)
    #     # select only trials for the corresponding label
    #     valid_trials_id = trial_df["trial_id"][
    #         (trial_df["labels"] == fit) & (trial_df["bad"] == 0)
    #     ].values

    # if MAX_TRIALS < X.shape[0]:
    # if MAX_TRIALS < len(valid_trials_id):
    #     if RANDOM_TRIAL:
    #         idx = np.random.choice(valid_trials_id, MAX_TRIALS, replace=False)
    #     else:
    #         idx = valid_trials_id[:MAX_TRIALS]
    #     X = X[idx]
    # else:
    #     idx = valid_trials_id
    #     X = X[valid_trials_id]

    X_train, X_test, idx = get_train_test_data(
        subject_id,
        n_trials_train=n_trials_train,
        n_trials_test=n_trials_test,
        q=q,
        l_freq=l_freq,
        h_freq=h_freq,
    )

    exp_params.update(
        q=q,
        l_freq=l_freq,
        h_freq=h_freq,
    )
    exp_dir = Path(f"benchmark_apnea-ecg/{subject_id}/{hash(exp_params)}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(exp_dir)

    # save experiment parameters
    with open(exp_dir / "exp_params.json", "w") as outfile:
        json.dump(exp_params, outfile, indent=4)

    subject_dir = exp_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    # compute random dict and save
    # d_random = init_dictionary(
    #     X_train,
    #     cdl_params["n_atoms"],
    #     cdl_params["n_times_atom"],
    #     D_init="random",
    #     rank1=False,
    #     random_state=cdl_params["random_state"],
    # )
    # np.save(subject_dir / f"d_random_{fit}", d_random)

    # compute chunk dict and save
    # d_chunk = init_dictionary(
    #     X_train,
    #     cdl_params["n_atoms"],
    #     cdl_params["n_times_atom"],
    #     D_init="chunk",
    #     rank1=False,
    #     random_state=cdl_params["random_state"],
    # )
    # np.save(subject_dir / f"d_chunk_{fit}", d_chunk)

    # set init dictionary
    # if cdl_params["D_init"] == "random":
    #     cdl_params["D_init"] = d_random
    # elif cdl_params["D_init"] == "chunk":
    #     cdl_params["D_init"] = d_chunk

    cdl_params["D_init"] = init_dictionary(
        X_train,
        cdl_params["n_atoms"],
        cdl_params["n_times_atom"],
        D_init=cdl_params["D_init"],
        rank1=False,
        random_state=cdl_params["random_state"],
    )

    # np.save(subject_dir / f"d_init_{fit}", cdl_params["D_init"])

    # fit CDL model
    pobj, times, d_hat, z_hat, reg = run_cdl(X_train, cdl_params)
    # compute loss on test set
    z_hat_test, _, _ = update_z_multi(
        X_test,
        d_hat,
        reg=reg,
        solver=cdl_params["solver_z"],
        solver_kwargs=cdl_params["solver_z_kwargs"],
        n_jobs=cdl_params["n_jobs"],
    )
    cost_test = compute_X_and_objective_multi(
        X_test,
        z_hat_test,
        D_hat=d_hat,
        reg=None,
        feasible_evaluation=True,
        uv_constraint="joint",
        return_X_hat=False,
    )

    # save results
    np.save(subject_dir / f"pobj_{fit}{suffix}", pobj)
    np.save(subject_dir / f"times_{fit}{suffix}", times)
    np.save(subject_dir / f"z_hat_{fit}{suffix}", z_hat)
    np.save(subject_dir / f"d_hat_{fit}{suffix}", d_hat)
    # save global res dictionary
    dict_res = dict(
        subject_id=subject_id,
        cost_init=pobj[0],
        cost=pobj[-1],
        cost_test=cost_test,
        compute_time=np.cumsum(times)[-1],
        reg=reg,
        # idx=idx,
        q=q,
        l_freq=l_freq,
        h_freq=h_freq,
    )
    with open(subject_dir / f"dict_res_{fit}{suffix}.pkl", "wb") as fp:
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


# %%

df_b01_global = pd.DataFrame()
for q in [None, 1, 2.5]:
    for l_freq, h_freq in [(None, None), (0.5, 20)]:
        for seed in tqdm(range(5)):
            cdl_params["random_state"] = seed
            dict_res = proc(
                "b01",
                cdl_params,
                n_trials_train=20,
                n_trials_test=0,
                q=q,
                l_freq=l_freq,
                h_freq=h_freq,
                suffix=f"_{seed}",
            )
            dict_res["seed"] = seed
            this_df = pd.DataFrame([dict_res])
            df_b01_global = pd.concat([df_b01_global, this_df], ignore_index=True)
            df_b01_global.to_csv("apnea-ecg/df_b01_global.tsv", sep="\t", index=False)


1 / 0


data = []
for subject_id in tqdm(subject_id_list):
    # for subject_id in tqdm(["b01"]):
    dict_res = proc(subject_id, cdl_params)
    data.append(dict_res)

    df_cost = pd.DataFrame(data=data)
    name = "df_cost_self"
    if group_id is not None:
        name += f"_{group_id}"
    else:
        name += "_all"
    name += f"_{fit}_{n_iter_fista}_{reg}.csv"
    df_cost.to_csv(name, index=False)
# %%
