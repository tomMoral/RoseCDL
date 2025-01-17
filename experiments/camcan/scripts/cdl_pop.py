"""
For a subject, what is the loss obtained with the global population dict, the
dict of their age category (and the other categories), the dict of a subject of
their age category (and from the other catagories), compared to the one obtained
with their own dict.
"""
# %%
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from joblib import Parallel, delayed


from experiments.scripts.utils import get_lambda_global, get_D_sub, get_subject_z_and_cost


BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

N_ATOMS = 70
N_TIMES_ATOM = 150

N_JOBS = 40
DEVICE = "cuda:1"


# n_subjects = 10
# list_subjects_path = SUBJECTS_PATH[:n_subjects]

with open('./results/dict_dataset_cat1.pickle', 'rb') as handle:
    dict_dataset_cat1 = pickle.load(handle)

list_subjects_path = dict_dataset_cat1['test']
n_subjects = len(list_subjects_path)
n_jobs = 5

LMBD, list_lmbd = get_lambda_global(
    list_subjects_path, N_ATOMS, N_TIMES_ATOM, reg=0.3, method=np.median)

# %%


def get_D_cat(cat):
    """

    """
    # DICT_CAT_PATH = BASE_PATH / f"cdl-population/results/cat{cat}"
    # u_all = np.load(DICT_CAT_PATH / f'u_pruned_cat{cat}.npy')
    # v_all = np.load(DICT_CAT_PATH / f'v_pruned_cat{cat}.npy')
    # D_cat = np.c_[u_all, v_all]
    D_cat = np.load(f'../results/D_hat_cat{cat}.npy')
    return D_cat


# %%


def my_func(i, list_subjects_path):

    subject_path = list_subjects_path[i]
    subject_id = subject_path.name.split('.')[0]

    # compute the subject dict
    D_init, D_hat_sub, lmbd = get_D_sub(
        subject_path, n_atoms=N_ATOMS, n_times_atom=N_TIMES_ATOM, lmbd=0.1)

    results = Parallel(n_jobs=n_jobs)(
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


def get_df_cost_cat(cat, list_subjects_path):
    D_cat = get_D_cat(cat=cat)
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_subject_z_and_cost)(
            subject_path, D_cat, reg=LMBD)
        for subject_path in list_subjects_path)

    dic_res = []
    for res in results:
        dic_res.append(dict(
            subject_id=res[0],
            dict_fit=f'D_cat{cat}',
            cost=res[2]
        ))

    df_cost_cat = pd.DataFrame(data=dic_res)
    df_cost_cat.to_csv('df_cost_cat1.csv')
    df_cost_cat['subject_id'] = df_cost_cat['subject_id'].apply(
        lambda x: x[4:])

    return df_cost_cat


df_cost_cat1 = get_df_cost_cat(1, list_subjects_path)
# df_cost_cat2 = get_df_cost_cat(2, list_subjects_path)


# get global dict
u_all = np.load(DICT_PATH / 'u_all_camcan.npy')
v_all = np.load(DICT_PATH / 'v_all_camcan.npy')
D_all = np.c_[u_all, v_all]

results = Parallel(n_jobs=n_jobs)(
    delayed(get_subject_z_and_cost)(
        subject_path, D_all, reg=LMBD)
    for subject_path in list_subjects_path)

dic_res = []
for res in results:
    dic_res.append(dict(
        subject_id=res[0],
        dict_fit='D_all',
        cost=res[2]
    ))

df_cost_all = pd.DataFrame(data=dic_res)
df_cost_all.to_csv('df_cost_all.csv')
df_cost_all['subject_id'] = df_cost_all['subject_id'].apply(lambda x: x[4:])


# %%
# df_cost = pd.read_csv('df_cost.csv')
# df_cost['subject_id'] = df_cost['subject_id'].apply(lambda x: x[4:])

# # get sub-dataframe for D_init
# d_init_index = df_cost[df_cost['dict_fit'] == 'D_init'].index
# df_init = df_cost.loc[d_init_index.values]
# df_init['cost'] = df_init['cost'].apply(
#     lambda x: float(x.split(',')[-1][1:-1]))

# df_cost.drop(d_init_index.values, inplace=True)
# df_cost['cost'] = df_cost['cost'].astype(float)
# df_cost['dict_fit'] = df_cost['dict_fit'].apply(lambda x: x[4:])

# # get sub-dataframe for D_self
# self_index = df_cost[df_cost['subject_id'] == df_cost['dict_fit']].index
# df_self = df_cost.loc[self_index.values]

# df_cost.drop(self_index.values, inplace=True)


# df_boxplot = df_cost[df_cost['subject_id'] != df_cost['dict_fit']]
# g = sns.boxplot(data=df_cost, x="subject_id", y="cost")

# xticklabels = [t.get_text() for t in g.get_xticklabels()]

# yy_self = [df_self[df_self['subject_id'] == xlabel]['cost'].values[0]
#            for xlabel in xticklabels]
# plt.scatter(xticklabels, yy_self, marker='*', label='self')

# yy_init = [df_init[df_init['subject_id'] == xlabel]['cost'].values[0]
#            for xlabel in xticklabels]
# plt.scatter(xticklabels, yy_init, marker='v', label='init')

# for i, df_cat in enumerate([df_cost_cat1, df_cost_cat2]):
#     yy_cat = [df_cat[df_cat['subject_id'] == xlabel]['cost'].values[0]
#               for xlabel in xticklabels]
#     plt.scatter(xticklabels, yy_cat, marker='o', label=f'cat{i+1}', alpha=0.5)

# yy_all = [df_cost_all[df_cost_all['subject_id'] == xlabel]['cost'].values[0]
#           for xlabel in xticklabels]
# plt.scatter(xticklabels, yy_all, marker='P', label='all', alpha=0.5)

# plt.xticks(rotation=90)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

# %%


# %%
