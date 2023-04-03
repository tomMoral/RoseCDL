# %%
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

from experiments.scripts.utils import get_lambda_global

BASE_PATH = Path('/storage/store2/work/bmalezie')
DICT_PATH = BASE_PATH / 'cdl-population/results/camcan'
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

N_ATOMS = 40
N_TIMES_ATOM = 150

n_subjects = 20
list_subjects_path = SUBJECTS_PATH[:n_subjects]
n_rep = 50

# %%


def proc():
    LMBD, list_lmbd = get_lambda_global(
        list_subjects_path, N_ATOMS, N_TIMES_ATOM, q=0.9, reg=1,
        method=np.median)

    row = [dict(subject_id=sub.name.split('.')[0][4:], lmbd_max=lmbd)
           for sub, lmbd in zip(list_subjects_path, list_lmbd)]

    return row


results = Parallel(n_jobs=20, verbose=10)(
    delayed(proc)()
    for _ in range(n_rep))
# %%
data = [dic for res in results for dic in res]
df_lmbd = pd.DataFrame(data=data)

lmbd_global = df_lmbd["lmbd_max"].median()
g = sns.boxplot(data=df_lmbd, x="subject_id", y="lmbd_max")
g.axhline(lmbd_global, label=f'median global lambda max: {lmbd_global:.2f}')
plt.xticks(rotation=90)
plt.legend(loc='best')
plt.title(f"Lamba max of each subjects over {n_rep} repetitions")
plt.tight_layout()
plt.savefig('exp_lambda_global.pdf', dpi=300)
plt.show()
# %%
