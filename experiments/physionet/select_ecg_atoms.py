# %%
import numpy as np
from scipy.signal import tukey

import matplotlib.pyplot as plt
from alphacsc.init_dict import init_dictionary

from utils_apnea import load_ecg, plot_temporal_atoms, run_cdl


subject_id = "b01"

X, labels = load_ecg(
    subject_id,
    split=True,
    T=60,
    apply_window=True,
    l_freq=None,
    h_freq=None,
    verbose=False,
)
n_times_atom = 75

ds = []
ds.append(X[0][0][260 : 260 + n_times_atom])  # b01
ds.append(X[0][0][3430 : 3430 + n_times_atom])  # b01
ds.append(X[447][0][1500 : 1500 + n_times_atom])
ds *= tukey(n_times_atom, alpha=0.2)[None, :]
ds = np.array(ds)
np.save("atoms_ecg", ds)

plot_temporal_atoms(ds)

xx = np.arange(0, n_times_atom) * 10
for kk, atom in enumerate(ds):
    plt.plot(xx, atom, alpha=0.9, label=f"atom {kk}")
plt.title(f"Selection of {kk+1} realistic ECG atoms from subject {subject_id}")
plt.xlabel("Time (ms.)")
plt.ylabel("Temporal")
plt.xlim(0, (n_times_atom - 1) * 10)
plt.legend()
plt.show()


# %%
