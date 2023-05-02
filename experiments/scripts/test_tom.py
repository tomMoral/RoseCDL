# %%
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

from alphacsc.utils.convolution import construct_X_multi
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
    # add mean
    m = X.mean(axis=1)
    data = dict(subject_id=np.repeat(subject_id, n_channels), value=m,
                type='mean', normalized=False)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])
    # add std
    std = X.std(axis=1)
    data = dict(subject_id=np.repeat(
        subject_id, n_channels), value=std, type='std', normalized=False)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])

    # apply normlization
    X -= X.mean()
    # X -= X.mean(axis=1, keepdims=True)
    # X /= X.std()
    n_times_atom = 150
    patch = np.ones(shape=n_times_atom)
    var_patch = np.sum([np.convolve(patch, diff_i, mode='valid')
                        for diff_i in X**2], axis=0) / n_times_atom
    var_patch -= (np.sum([np.convolve(patch, diff_i, mode='valid')
                          for diff_i in X], axis=0)/n_times_atom)**2
    var_patch = var_patch.clip(0)

    X /= np.sqrt(np.median(var_patch))
    # X /= X.std(axis=1, keepdims=True)
    # add mean
    m = X.mean(axis=1)
    data = dict(subject_id=np.repeat(subject_id, n_channels), value=m,
                type='mean', normalized=True)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])
    # add std
    std = X.std(axis=1)
    data = dict(subject_id=np.repeat(
        subject_id, n_channels), value=std, type='std', normalized=True)
    df_summary = pd.concat([df_summary, pd.DataFrame(data=data)])

# %%
# sns.boxplot(df_summary, x='subject_id', y='mean')
g = sns.catplot(
    data=df_summary, x="subject_id", y="value", row="type",
    col='normalized',
    kind="box", sharey=False
)
# g = sns.FacetGrid(df_summary, col="normalized", row="type", margin_titles=True, sharey=False)
# g.map_dataframe(sns.boxplot, x="subject_id", y="value")
g.set_axis_labels("Subject id", "Values across channels")
g.set_titles(row_template="{row_name}")
# g.tight_layout()
# plt.xticks(rotation=90)
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
plt.show()
# %%


X1 = np.load(DATA_PATH / 'cat1' / 'sub-CC110033.npy')
X1 -= X1.mean()
X2 = np.load(DATA_PATH / 'cat1' / 'sub-CC110037.npy')
X2 -= X2.mean()


def get_var_patch(X, clip_value=None):
    patch = np.ones(shape=N_TIMES_ATOM)

    var_patch = np.sum([np.convolve(patch, diff_i, mode='valid')
                        for diff_i in X**2], axis=0) / N_TIMES_ATOM
    var_patch -= (np.sum([np.convolve(patch, diff_i, mode='valid')
                          for diff_i in X], axis=0) / N_TIMES_ATOM)**2

    if clip_value is not None:
        return var_patch.clip(clip_value)

    return var_patch


def apply_quantile(a, q):
    """

    Parameters
    ----------
    a : np.array

    q : float, between 0 and 1

    Returns
    -------
    np.array
    """
    quantile = np.quantile(a, q)
    print(f"{q} quantile: {quantile}")
    return a[a <= quantile]


def plot_var_patch(list_var_patch, labels, title, q=None):
    if q is not None:
        list_var_patch = [apply_quantile(this_var_patch, q)
                          for this_var_patch in list_var_patch]

    for this_var_patch, this_label in zip(list_var_patch, labels):
        plt.hist(this_var_patch, density=True, bins=50,
                 alpha=0.3, label=this_label)

    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()

    if q is not None:
        return list_var_patch


var_patch_1 = get_var_patch(X1, clip_value=0)
print('var_patch_1:', var_patch_1)
print('variace:', X1.var())
# %%


var_patch_2 = get_var_patch(X2, clip_value=0)
list_var_patch = plot_var_patch(
    [var_patch_1, var_patch_2], ['CC110033', 'CC110033'],
    title="Patches variances before scaling")

X1 /= np.sqrt(np.median(list_var_patch[0]))
X2 /= np.sqrt(np.median(list_var_patch[1]))

var_patch_1 = get_var_patch(X1, clip_value=0)
var_patch_2 = get_var_patch(X2, clip_value=0)
plot_var_patch([var_patch_1, var_patch_2], ['CC110033', 'CC110037'],
               title="Patches variances after scaling", q=0.9)
# %%
