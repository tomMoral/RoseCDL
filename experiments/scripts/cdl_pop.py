"""
For a subject, what is the loss obtained with the global population dict, the
dict of their age category (and the other categories), the dict of a subject of
their age category (and from the other catagories), compared to the one obtained
with their own dict.
"""
# %%
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from joblib import Memory, Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

from alphacsc.update_z_multi import update_z_multi
from alphacsc.loss_and_gradient import compute_X_and_objective_multi
from alphacsc.init_dict import init_dictionary
from alphacsc.utils.dictionary import get_lambda_max

from wincdl.wincdl import WinCDL


BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

N_ATOMS = 40
N_TIMES_ATOM = 150

N_JOBS = 40
DEVICE = "cuda:1"


# get a unique value for lambda
X = np.load(SUBJECTS_PATH[0])
X /= X.std()
if X.ndim == 2:
    (n_channels, n_times) = X.shape
    n_trials = 1
elif X.ndim == 3:
    (n_trials, n_channels, n_times) = X.shape

# get initial dictionary with alphacsc
D_init = init_dictionary(
    X[None, :], N_ATOMS, N_TIMES_ATOM, uv_constraint='separate',
    rank1=True, window=True, D_init='chunk', random_state=None)
LMBD = 0.1 * get_lambda_max(X[None, :], D_init).max()
# %%


def get_D_cat(cat):
    """

    """
    DICT_CAT_PATH = BASE_PATH / f"cdl-population/results/cat{cat}"
    u_all = np.load(DICT_CAT_PATH / f'u_pruned_cat{cat}.npy')
    v_all = np.load(DICT_CAT_PATH / f'v_pruned_cat{cat}.npy')
    D_cat = np.c_[u_all, v_all]
    return D_cat


def get_D_sub(subject_path, n_atoms=40, n_times_atom=150, lmbd=0.1):
    """Get subject's self dictionary using Windowing-CDL
    """

    X = np.load(subject_path)
    X /= X.std()
    if X.ndim == 2:
        (n_channels, n_times) = X.shape
        n_trials = 1
    elif X.ndim == 3:
        (n_trials, n_channels, n_times) = X.shape

    # get initial dictionary with alphacsc
    D_init = init_dictionary(
        X[None, :], n_atoms, n_times_atom, uv_constraint='separate',
        rank1=True, window=True, D_init='chunk', random_state=None)
    # lmbd = lmbd * get_lambda_max(X[None, :], D_init).max()

    CDL = WinCDL(
        n_components=n_atoms,
        kernel_size=n_times_atom,
        n_channels=n_channels,
        lmbd=LMBD,
        n_iterations=30,
        epochs=50,
        max_batch=5,
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

    CDL.fit(X)

    return D_init, CDL.D_hat_, lmbd


def get_subject_z_and_cost(subject_path, uv_hat_, reg=0.1, tt_max=None):
    """Compute subject sparse vector for a given dictionnary and pre-processed
    signal.

    Parameters
    ----------
    subject_path : Pathlib instance
        path to subject's pre-processed raw signal as numpy array,
        must end by [subject_id].npy

    uv_hat_ : array-like, shape (n_atoms, n_channels + n_times_atom)
        learned atoms' dictionnary

    reg : float
        value for sparsity regularization

    tt_max : int | None
        if int, only the first `tt_max` indices will be considered
        if None, the whole signal is considered

    Returns
    -------
    subject_dict : dict
        subject_id, age, sex : subject infos
        z_hat : 2d-array shape (n_atoms, n_times)
        n_acti : 1d-array of length n_atoms
            number of non-null activations for each atom
    """

    subject_id = subject_path.name.split('.')[0]

    X = np.load(subject_path)
    X /= X.std()
    if tt_max is not None:
        X = X[:, :tt_max]
    # compute sparse vector z
    z_hat, _, _ = update_z_multi(
        X[None, :],
        uv_hat_.astype(np.float64), reg=reg,
        solver='lgcd', solver_kwargs={'tol': 1e-3, 'max_iter': 10_000},
        n_jobs=1
    )
    # compute associated cost
    cost = compute_X_and_objective_multi(
        X[None, :], z_hat, D_hat=uv_hat_, reg=reg,
        uv_constraint='separate')

    return subject_id, z_hat, cost


# %%


dic_res = []
n_subjects = 10
list_subjects_path = SUBJECTS_PATH[:n_subjects]


def my_func(i, list_subjects_path):

    subject_path = list_subjects_path[i]
    subject_id = subject_path.name.split('.')[0]

    # compute the subject dict
    D_init, D_hat_sub, lmbd = get_D_sub(
        subject_path, n_atoms=N_ATOMS, n_times_atom=N_TIMES_ATOM, lmbd=0.1)

    results = Parallel(n_jobs=n_subjects)(
        delayed(get_subject_z_and_cost)(
            subject_path, D_hat_sub, reg=LMBD)
        for subject_path in list_subjects_path)

    dic_res = []
    for res in results:
        dic_res.append(dict(
            subject_id=res[0],
            dict_fit=subject_id,
            cost=res[2]
        ))

    cost_init = get_subject_z_and_cost(subject_path, D_init, reg=LMBD)[2]
    dic_res.append(dict(
        subject_id=subject_id,
        dict_fit='D_init',
        cost=cost_init
    ))

    return dic_res


results = [my_func(i, list_subjects_path) for i in range(n_subjects)]
data = []
for dic_res in results:
    for this_dict in dic_res:
        data.append(this_dict)

df_cost = pd.DataFrame(data=data)
df_cost.to_csv('df_cost.csv')

# %%
df_cost = pd.read_csv('df_cost.csv')
df_cost['subject_id'] = df_cost['subject_id'].apply(lambda x: x[4:])

# get sub-dataframe for D_init
d_init_index = df_cost[df_cost['dict_fit'] == 'D_init'].index
df_init = df_cost.loc[d_init_index.values]
df_init['cost'] = df_init['cost'].apply(
    lambda x: float(x.split(',')[-1][1:-1]))

df_cost.drop(d_init_index.values, inplace=True)
df_cost['cost'] = df_cost['cost'].astype(float)
df_cost['dict_fit'] = df_cost['dict_fit'].apply(lambda x: x[4:])

# get sub-dataframe for D_self
self_index = df_cost[df_cost['subject_id'] == df_cost['dict_fit']].index
df_self = df_cost.loc[self_index.values]

df_cost.drop(self_index.values, inplace=True)


df_boxplot = df_cost[df_cost['subject_id'] != df_cost['dict_fit']]
g = sns.boxplot(data=df_cost, x="subject_id", y="cost")

xticklabels = [t.get_text() for t in g.get_xticklabels()]

yy_self = [df_self[df_self['subject_id'] == xlabel]['cost'].values[0]
           for xlabel in xticklabels]
plt.scatter(xticklabels, yy_self, marker='*', label='self')

yy_init = [df_init[df_init['subject_id'] == xlabel]['cost'].values[0]
           for xlabel in xticklabels]
plt.scatter(xticklabels, yy_init, marker='v', label='init')

plt.xticks(rotation=90)
plt.legend()
plt.show()

# %%
# get global dict
u_all = np.load(DICT_PATH / 'u_all_camcan.npy')
v_all = np.load(DICT_PATH / 'v_all_camcan.npy')
D_all = np.c_[u_all, v_all]

D_cat1 = get_D_cat(cat=1)

dic_res = []
for i, subject_path in tqdm(enumerate(list_subjects_path)):
    subject_id = subject_path.name.split('.')[0]
    # compute cost with the global dict
    # _, _, cost_all = get_subject_z_and_cost(
    #     subject_path, D_all, reg=LMBD)
    # dic_res.append(dict(
    #     subject_id=subject_id,
    #     dict_fit='D_all',
    #     cost=cost_all
    # ))
    # compute cost with the category 1 dict
    _, _, cost_cat1 = get_subject_z_and_cost(
        subject_path, D_cat1, reg=LMBD)
    dic_res.append(dict(
        subject_id=subject_id,
        dict_fit='D_cat1',
        cost=cost_cat1
    ))


df_res = pd.DataFrame(data=dic_res)
df_res.to_csv('df_res.csv')

# %%
