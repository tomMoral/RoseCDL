"""From alphacsc results, get the multitude of outputed reg values and compute
a general value to put as hyperparameter for WinCDL"""

# %%
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm

group_id = "a"
fit = "N"

subject_id_list = pd.read_csv(Path("apnea-ecg/participants.tsv"), sep="\t")[
    "Record"
].values
group_des = dict(a="apnea", b="borderline apnea", c="control", x="test")

if group_id is not None:
    subject_id_list = [id for id in subject_id_list if id[0] == group_id]

# subject_id_list = subject_id_list[:3]
for subject_id in tqdm(subject_id_list):
    subject_dir = Path(f"apnea-ecg/{subject_id}")
    with open(subject_dir / f"dict_res_{fit}.pkl", "rb") as f:
        dict_res = pickle.load(f)
        reg = dict_res["reg"]
        print(reg)

# %%
