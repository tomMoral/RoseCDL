# %%
import pickle

import torch
import yaml
from alphacsc import BatchCDL, GreedyCDL, OnlineCDL

# from compute_dict_similarity import recovery_score
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


# Load the configuration file
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
BATCH_SIZE = config["data"]["batch_size"]

# %%
# ======== CALLBACK ========
D_batch = []
D_greedy = []
D_online = []


def callback_batch(z_encoder, pobj):
    D_hat = z_encoder.D_hat
    D_batch.append(D_hat)


def callback_greedy(z_encoder, pobj):
    D_hat = z_encoder.D_hat
    D_greedy.append(D_hat)


def callback_online(z_encoder, pobj):
    D_hat = z_encoder.D_hat
    D_online.append(D_hat)


# %%
# ======== SUPER LISTS ========
super_list_D = []
super_D_batch = []
super_D_greedy = []
super_D_online = []

# ======== TRAINING ========
for run in range(N_RUNS):
    print(f"Run {run + 1}/{N_RUNS}")

    # Reset lists for each run
    D_batch = []
    D_greedy = []
    D_online = []

    # ======== DATA SIMULATION ========
    X, ds_true, z_true = simulate_data(n_trials=config["data"]["n_trials"])

    # ======== DATALOADER CREATION ========
    dataset = SignalDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ======== WINCDL CREATION ========
    wincdl = WinCDL(
        n_components=N_COMP,
        kernel_size=KERNEL_SIZE,
        n_channels=N_CHANNELS,
        lmbd=LMBD,
        n_iterations=N_ITER,
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

    batchcdl.callback = callback_batch
    greedycdl.callback = callback_greedy
    onlinecdl.callback = callback_online

    # Train the models
    losses, list_D, times = wincdl.fit(dataloader)
    batchcdl.fit(X)
    greedycdl.fit(X)
    onlinecdl.fit(X)

    # Add results to super lists
    super_list_D.append(list_D)
    super_D_batch.append(D_batch)
    super_D_greedy.append(D_greedy)
    super_D_online.append(D_online)

# ======== SAVING ========
with open("super_list_D.pkl", "wb") as f:
    pickle.dump(super_list_D, f)
with open("super_D_batch.pkl", "wb") as f:
    pickle.dump(super_D_batch, f)
with open("super_D_greedy.pkl", "wb") as f:
    pickle.dump(super_D_greedy, f)
with open("super_D_online.pkl", "wb") as f:
    pickle.dump(super_D_online, f)
with open("ds_true.pkl", "wb") as f:
    pickle.dump(ds_true, f)


# # %%
# print("WINCDL :", list_D[-1].shape, all([D.shape == list_D[-1].shape for D in list_D]))
# print(
#     "BATCHCDL :",
#     D_batch[-1].shape,
#     all([D.shape == D_batch[-1].shape for D in D_batch]),
# )
# print(
#     "GREEDYCDL :",
#     D_greedy[-1].shape,
#     all([D.shape == D_greedy[-1].shape for D in D_greedy]),
# )
# print(
#     "ONLINECDL :",
#     D_online[-1].shape,
#     all([D.shape == D_online[-1].shape for D in D_online]),
# )
# print("TRUE :", ds_true.shape)

# win_recovery_scores = [recovery_score(D.squeeze(1), ds_true.squeeze(1)) for D in list_D]
# batch_recovery_score = [recovery_score(D, ds_true) for D in D_batch]

# greedy_recovery_score = [
#     recovery_score(D.squeeze(1), ds_true.squeeze(1)) for D in D_greedy
# ]
# online_recovery_score = [
#     recovery_score(D.squeeze(1), ds_true.squeeze(1)) for D in D_online
# ]

# print("WINCDL recovery score", win_recovery_scores[-1])
# print("BATCHCDL recovery score", batch_recovery_score[-1])
# print("GREEDYCDL recovery score", greedy_recovery_score[-1])
# print("ONLINECDL recovery score", online_recovery_score[-1])
# # %%
# plt.figure()
# plt.scatter(0, win_recovery_scores[-1], label="WinCDL")
# plt.scatter(0, batch_recovery_score[-1], label="BatchCDL")
# plt.scatter(0, greedy_recovery_score[-1], label="GreedyCDL")
# plt.scatter(0, online_recovery_score[-1], label="OnlineCDL")
# plt.legend()
# plt.ylabel("Recovery score")
# plt.savefig("last_recov.png")

# %%
