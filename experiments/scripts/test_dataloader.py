# %%
from pathlib import Path
import torch

from wincdl.datasets import PhysionetDataset, create_physionet_dataloader
from wincdl.wincdl import WinCDL


apnea_dataloader = create_physionet_dataloader(
    db_dir='../physionet/apnea-ecg', group_id='a')
apnea_cdl = WinCDL(
    n_components=10,
    kernel_size=75,
    n_channels=1,
    lmbd=0.1,
    epochs=5,
    device='cuda:1',
    list_D=True,
)
losses, list_D, times = apnea_cdl.fit(apnea_dataloader)
# %%
