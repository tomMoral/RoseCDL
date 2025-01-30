import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from utils_apnea import plot_multi_subject_temporal_atoms

parser = argparse.ArgumentParser()
parser.add_argument(
    "--group",
    type=str,
    choices=["a", "b", "c", "x"],
    help="group id to run the CDL on ('a': apnea, 'b': borderline apnea, 'c': control, 'x': test)",
)
parser.add_argument("--fit", type=str, default="N")

args = parser.parse_args()
group_id = args.group
fit_on = args.fit

subject_id_list = pd.read_csv(Path("apnea-ecg/participants.tsv"), sep="\t")[
    "Record"
].values
group_des = dict(a="apnea", b="borderline apnea", c="control", x="test")

if group_id is not None:
    subject_id_list = [id for id in subject_id_list if id[0] == group_id]
    print(f"Plot all atoms of group {group_id} ({group_des[group_id]})")
else:
    print("Plot all atoms")

dict_d_hat = {}
for subject_id in subject_id_list:
    subject_dir = Path(f"apnea-ecg/{subject_id}")
    subject_d_hat = np.load(subject_dir / f"d_hat_{fit_on}.npy")
    dict_d_hat[subject_id] = subject_d_hat

plot_multi_subject_temporal_atoms(
    dict_d_hat, save_fig=f"./group_{group_id}_{fit_on}_atoms.pdf"
)
