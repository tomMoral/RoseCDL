# %%
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

from alphacsc.utils.convolution import construct_X_multi
from alphacsc.init_dict import init_dictionary
from experiments.scripts.utils import get_lambda_global

BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

N_ATOMS = 40
N_TIMES_ATOM = 150

n_subjects = 20
list_subjects_path = SUBJECTS_PATH[:n_subjects]


df_summary = pd.DataFrame()
for sub_path in list_subjects_path:
    subject_id = sub_path.name.split('.')[0][4:]
    X = np.load(sub_path)
    n_channels = X.shape[0]

    patch = np.ones(shape=N_TIMES_ATOM)
    # add patches mean
    mean_patches = (np.sum([np.convolve(patch, diff_i, mode='valid')
                            for diff_i in X], axis=0)/N_TIMES_ATOM)
    n_patches = mean_patches.shape[0]
    data = dict(subject_id=np.repeat(subject_id, n_patches),
                value=mean_patches, type='mean', normalized=False)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])
    # add variances
    var_patch = np.sum([np.convolve(patch, diff_i, mode='valid')
                        for diff_i in X**2], axis=0) / N_TIMES_ATOM
    var_patch -= mean_patches**2
    data = dict(subject_id=np.repeat(subject_id, n_patches),
                value=var_patch, type='variances', normalized=False)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])

    # apply normlization
    X -= X.mean()
    # X /= X.std()
    patch = np.ones(shape=N_TIMES_ATOM)
    var_patch = np.sum([np.convolve(patch, diff_i, mode='valid')
                        for diff_i in X**2], axis=0) / N_TIMES_ATOM
    var_patch -= (np.sum([np.convolve(patch, diff_i, mode='valid')
                          for diff_i in X], axis=0)/N_TIMES_ATOM)**2
    var_patch = var_patch.clip(0)

    X /= np.sqrt(np.median(var_patch))

    # add patches mean
    mean_patches = (np.sum([np.convolve(patch, diff_i, mode='valid')
                            for diff_i in X], axis=0)/N_TIMES_ATOM)
    n_patches = mean_patches.shape[0]
    data = dict(subject_id=np.repeat(subject_id, n_patches),
                value=mean_patches, type='mean', normalized=True)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])
    # add variances
    var_patch = np.sum([np.convolve(patch, diff_i, mode='valid')
                        for diff_i in X**2], axis=0) / N_TIMES_ATOM
    var_patch -= mean_patches**2
    data = dict(subject_id=np.repeat(subject_id, n_patches),
                value=var_patch, type='variances', normalized=True)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])

# %%
g = sns.catplot(
    data=df_summary, x="subject_id", y="value", row="type",
    col='normalized',
    kind="violin", sharey=False,  # whis=5,
)
# g = sns.FacetGrid(df_summary, col="normalized", row="type", margin_titles=True, sharey=False)
# g.map_dataframe(sns.boxplot, x="subject_id", y="value")
g.set_axis_labels("Subject id", "Values across patches")
g.set_titles(row_template="{row_name}")
# g.tight_layout()
# plt.xticks(rotation=90)
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
plt.show()
# %%

df_summary = pd.DataFrame()

for sub_path in list_subjects_path:
    subject_id = sub_path.name.split('.')[0][4:]
    X = np.load(sub_path)
    X = X[None, :]
    n_trials, n_channels, n_times = X.shape
    sample_weights = np.ones(n_trials)

    D_hat = init_dictionary(
        X, N_ATOMS, N_TIMES_ATOM, uv_constraint='separate',
        rank1=True, window=True, D_init='chunk', random_state=42)

    values = [[
        np.convolve(
            np.dot(uv_k[:n_channels], X_i * W_i), uv_k[:n_channels - 1:-1],
            mode='valid') for X_i, W_i in zip(X, sample_weights)
    ] for uv_k in D_hat]
    # values' shape : (40, 1, 81001)
    maxs = np.max(values, axis=(1, 2))
    data = dict(subject_id=np.repeat(subject_id, N_ATOMS),
                value=maxs, type='max', normalized=False)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])
    qauntile = np.quantile(values, axis=(1, 2), q=0.9)
    data = dict(subject_id=np.repeat(subject_id, N_ATOMS),
                value=qauntile, type='quantile', normalized=False)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])

    # apply normlization
    X = np.load(sub_path)
    X -= X.mean()
    # patch = np.ones(shape=N_TIMES_ATOM)
    # var_patch = np.sum([np.convolve(patch, diff_i, mode='valid')
    #                     for diff_i in X**2], axis=0) / N_TIMES_ATOM
    # var_patch -= (np.sum([np.convolve(patch, diff_i, mode='valid')
    #                       for diff_i in X], axis=0)/N_TIMES_ATOM)**2
    # var_patch = var_patch.clip(0)

    # X /= np.sqrt(np.median(var_patch))
    X /= X.std()
    X = X[None, :]

    D_hat = init_dictionary(
        X, N_ATOMS, N_TIMES_ATOM, uv_constraint='separate',
        rank1=True, window=True, D_init='chunk', random_state=42)

    values = [[
        np.convolve(
            np.dot(uv_k[:n_channels], X_i * W_i), uv_k[:n_channels - 1:-1],
            mode='valid') for X_i, W_i in zip(X, sample_weights)
    ] for uv_k in D_hat]
    # values' shape : (40, 1, 81001)
    maxs = np.max(values, axis=(1, 2))
    data = dict(subject_id=np.repeat(subject_id, N_ATOMS),
                value=maxs, type='max', normalized=True)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])
    qauntile = np.quantile(values, axis=(1, 2), q=0.8)
    data = dict(subject_id=np.repeat(subject_id, N_ATOMS),
                value=qauntile, type='quantile', normalized=True)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])
# %%
# g = sns.boxplot(data=df_summary, x="subject_id", y="value", )
# g.set(xlabel="Subject id", ylabel="Max lambda across atoms")
g = sns.catplot(
    data=df_summary, x="subject_id", y="value", row="type",
    col='normalized',
    kind="box", sharey=False
)
g.set_axis_labels("Subject id", "Values across atoms")
# plt.xticks(rotation=90)
# plt.legend(loc='best')
# plt.title(f"Lamba max of each subjects over the atoms")
# plt.tight_layout()
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
plt.savefig('stats_lambda_atoms.pdf', dpi=300)
plt.show()
# %%
