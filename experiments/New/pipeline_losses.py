"""In this script, we generate synthetic data and train alphacsc and
wincdl models on it. We save the losses at each iteration to plot the
evolution of the loss over iterations.
"""

import pickle

import matplotlib.pyplot as plt
import torch
import yaml
from alphacsc import BatchCDL, GreedyCDL, OnlineCDL
from simulate import simulate_data
from torch.utils.data import DataLoader, Dataset

from wincdl.wincdl import WinCDL


# ======== CUSTOM DATASET ========
class SignalDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Charger les paramètres de configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# ======== PARAMETERS ========
N_COMP = config["model"]["n_components"]
KERNEL_SIZE = config["model"]["kernel_size"]
N_CHANNELS = config["model"]["n_channels"]
LMBD = config["model"]["lambda"]
N_ITER = config["model"]["n_iterations"]
EPOCHS = config["model"]["epochs"]
N_RUNS = config["training"]["n_runs"]

super_losses_wincdl = []
super_losses_batchcdl = []
super_losses_greedycdl = []
super_losses_onlinecdl = []

for run in range(N_RUNS):
    print(f"Run {run + 1}/{N_RUNS}")

    # ======== DATA SIMULATION ========
    X, ds_true, z_true = simulate_data(n_trials=config["data"]["n_trials"])

    print("X", X.shape)

    # ======== DATALOADER CREATION ========
    dataset = SignalDataset(X)
    BATCH_SIZE = config["data"]["batch_size"]
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ======== WINCDL CREATION ========
    wincdl = WinCDL(
        n_components=N_COMP,
        kernel_size=KERNEL_SIZE,
        n_channels=N_CHANNELS,
        lmbd=LMBD,
        n_iterations=2 * N_ITER,
        epochs=EPOCHS,
        list_D=config["model"]["list_d"],
        window=config["model"]["window"],
    )

    # ======== BATCHCDL CREATION ========
    batchcdl = BatchCDL(
        n_atoms=N_COMP,
        n_times_atom=KERNEL_SIZE,
        n_iter=N_ITER,
        reg=LMBD,
        verbose=1,
    )

    # ======== GREEDYCDL CREATION ========
    greedycdl = GreedyCDL(
        n_atoms=N_COMP,
        n_times_atom=KERNEL_SIZE,
        n_iter=N_ITER,
        reg=LMBD,
        verbose=1,
    )

    # ======== ONLINECDL CREATION ========
    onlinecdl = OnlineCDL(
        n_atoms=N_COMP,
        n_times_atom=KERNEL_SIZE,
        n_iter=N_ITER,
        reg=LMBD,
        verbose=1,
    )

    # ======== FITTING ========
    losses, list_D, times = wincdl.fit(dataloader)
    batchcdl.fit(X)
    greedycdl.fit(X)
    onlinecdl.fit(X)

    super_losses_wincdl.append(losses)
    super_losses_batchcdl.append(batchcdl.pobj_)
    super_losses_greedycdl.append(greedycdl.pobj_)
    super_losses_onlinecdl.append(onlinecdl.pobj_)

# Sauvegarder les superlistes de pertes
with open("super_losses_wincdl.pkl", "wb") as f:
    pickle.dump(super_losses_wincdl, f)
with open("super_losses_batchcdl.pkl", "wb") as f:
    pickle.dump(super_losses_batchcdl, f)
with open("super_losses_greedycdl.pkl", "wb") as f:
    pickle.dump(super_losses_greedycdl, f)
with open("super_losses_onlinecdl.pkl", "wb") as f:
    pickle.dump(super_losses_onlinecdl, f)

plt.figure(figsize=(10, 5))
for losses in super_losses_wincdl:
    plt.plot(losses, label="WinCDL")
for losses in super_losses_batchcdl:
    plt.plot(losses, label="BatchCDL")
for losses in super_losses_greedycdl:
    plt.plot(losses, label="GreedyCDL")
for losses in super_losses_onlinecdl:
    plt.plot(losses, label="OnlineCDL")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss over iterations")
plt.legend()
plt.savefig("super_losses.png")
