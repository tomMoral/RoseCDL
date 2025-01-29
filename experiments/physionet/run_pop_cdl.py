# %%
import numpy as np

from utils_apnea import plot_temporal_atoms
from wincdl.datasets import create_physionet_dataloader
from wincdl.wincdl import WinCDL
# %%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--group",
    type=str,
    choices=["a", "b", "c", "x"],
    help="group id to run the CDL on ('a': apnea, 'b': borderline apnea, 'c': control, 'x': test)",
)
args = parser.parse_args()
group_id = args.group
print(f"Run population CDL on group {group_id}")
apnea_dataloader = create_physionet_dataloader(db_dir="./apnea-ecg", group_id=group_id)
apnea_cdl = WinCDL(
    n_components=10,
    kernel_size=75,
    n_channels=1,
    lmbd=0.1,
    n_iterations=100,
    epochs=50,
    max_batch=4,
    mini_batch_window=10_000,
    mini_batch_size=5,
    device="cuda:1",
    list_D=True,
)
losses, list_D, times = apnea_cdl.fit(apnea_dataloader)
np.save(f"d_hat_pop_{group_id}", apnea_cdl.D_hat_)
plot_temporal_atoms(apnea_cdl.D_hat_.squeeze(), save_fig=f"pop_{group_id}_atoms.pdf")
# %%
