# %%
import os
import numpy as np
from pathlib import Path
import pickle

from alphacsc.init_dict import init_dictionary

from wincdl.wincdl import WinCDL
from experiments.scripts.utils import get_lambda_global

DATA_PATH = Path("/storage/store2/work/bmalezie/camcan-cdl")

N_ATOMS = 70
N_TIMES_ATOM = 150

N_JOBS = 40
DEVICE = "cuda:1"

cat = os.listdir(DATA_PATH)[0]
print(f"Processing {cat}...")
data_path_cat = DATA_PATH / cat
# %%


# generator = torch.Generator()
# generator.manual_seed(2147483647)
# train_dataloader = torch.utils.data.DataLoader(
#     MEGPopDataset(
#         data_path_cat, window=1_000, n_samples=30, seed=100
#     ),
#     batch_size=10,
#     shuffle=True,
#     generator=generator
# )
# train_dataloader = create_conv_dataloader(
#     str(data_path_cat),
#     DEVICE,
#     dtype=torch.float,
#     mini_batch_size=10,
#     sto=False,
#     window=1_000,
#     random_state=2147483647,
#     dimN=1,
#     n_samples=30
# )

# subjects = train_dataloader.dataset.subjects
all_paths = [x for x in data_path_cat.glob('**/*') if x.is_file()]
# test_set = [x for x in all_paths if x not in subjects]

X = np.load(all_paths[0])
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
# %%
lmbd, list_lmbd_max = get_lambda_global(all_paths, N_ATOMS, N_TIMES_ATOM, reg=0.3,
                                        method=np.median)
# %%
CDL = WinCDL(
    n_components=N_ATOMS,
    kernel_size=N_TIMES_ATOM,
    n_channels=n_channels,
    lmbd=lmbd,
    n_iterations=30,
    epochs=50,
    max_batch=30,
    stochastic=False,
    optimizer="linesearch",
    lr=0.1,
    gamma=0.9,
    mini_batch_window=10_000,
    mini_batch_size=1,   # batch_size for the dataloader
    device=DEVICE,
    rank="uv_constraint",
    window=True,
    D_init=D_init,
    positive_z=True,
    list_D=False,
    dimN=1,
    n_samples=30
)

CDL.fit(str(data_path_cat))

dict_dataset = dict(train=CDL.subjects,
                    test=[x for x in all_paths if x not in CDL.subjects])
with open('./results/dict_dataset_cat1.pickle', 'wb') as handle:
    pickle.dump(dict_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

# CDL.scheduler = None
# losses, list_D, times = train(
#     CDL.csc,
#     train_dataloader,
#     CDL.optimizer,
#     torch.nn.MSELoss(),
#     scheduler=CDL.scheduler,
#     epochs=CDL.epochs,
#     max_batch=CDL.max_batch,
#     save_list_D=CDL.list_D,
#     stopping_criterion=not CDL.stochastic
# )

np.save('./results/D_hat_cat1', CDL.D_hat_)
# %%
