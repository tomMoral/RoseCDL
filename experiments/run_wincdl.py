import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from wincdl.datasets import SubwindowsDataset
from wincdl.datasets.simulated import simulate_1d as simulate_data2
from wincdl.wincdl import WinCDL

# ====== PARAMETERS ======
# Data
N_TRIALS = 100
N_TIMES = 500
N_TIMES_ATOM = 50
N_ATOMS = 5
WINDOW = False
OVERLAP = True
RANDOM_STATE = 42
P_ACTI = 0.7
P_CONTAMINATE = 0.1
BATCH_SIZE = 32

# Model
N_COMPONENTS = N_ATOMS
KERNEL_SIZE = N_TIMES_ATOM
N_CHANNELS = 1
LMBD = 0.1
N_ITERATIONS = 30
SAMPLE_WINDOW = 300
LIST_D = True


# ======== CUSTOM DATASET ========
class SignalDataset(Dataset):
    def __init__(self, data, sto=False, device="cpu", dtype=torch.float32, **kwargs):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ====== DATA ======
X, ds, _ = simulate_data2(
    n_trials=N_TRIALS,
    n_times=N_TIMES,
    n_times_atom=N_TIMES_ATOM,
    n_atoms=N_ATOMS,
    window=WINDOW,
    overlap=OVERLAP,
    random_state=RANDOM_STATE,
    p_acti=P_ACTI,
    p_contaminate=P_CONTAMINATE,
)
X = X[:, np.newaxis, :]
print("SHAPE X : ", X.shape)

X_dataset = SubwindowsDataset(
    X, sample_window=SAMPLE_WINDOW, device="cpu", dtype=torch.float32
)
loader = DataLoader(X_dataset, batch_size=BATCH_SIZE, shuffle=True)

losses = []


def callback(model, epoch, loss):
    losses.append(loss)


model = WinCDL(
    n_components=N_COMPONENTS,
    kernel_size=KERNEL_SIZE,
    n_channels=N_CHANNELS,
    lmbd=LMBD,
    n_iterations=N_ITERATIONS,
    epochs=40,
    window=WINDOW,
    sample_window=SAMPLE_WINDOW,
    callbacks=(callback,),
)

model.fit(X)

print("SHAPE D : ", model.D_hat_.shape)
plt.plot(losses)
plt.show()
