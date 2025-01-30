import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_apnea import get_subject_z_and_cost

parser = argparse.ArgumentParser()
parser.add_argument(
    "--group",
    type=str,
    choices=["a", "b", "c", "x"],
    help="group id to run the CDL on ('a': apnea, 'b': borderline apnea, 'c': control, 'x': test)",
)
args = parser.parse_args()
group_id = args.group

subject_id_list = pd.read_csv(Path("apnea-ecg/participants.tsv"), sep="\t")[
    "Record"
].values
group_des = dict(a="apnea", b="borderline apnea", c="control", x="test")

if group_id is not None:
    subject_id_list = [id for id in subject_id_list if id[0] == group_id]
    print(f"Get recovery DataFrame on group {group_id} ({group_des[group_id]})")
else:
    print("Get recovery DataFrame on all subjects")

# load population dictionary
d_hat_pop = np.load(f"d_hat_pop_{group_id}.npy").squeeze()

subjects_rows = []
for subject_id in tqdm(subject_id_list):
    _, cost_pop = get_subject_z_and_cost(subject_id, d_hat_pop)
    subjects_rows.append(dict(subject_id=subject_id, dict_fit="D_pop", cost=cost_pop))

recovery_df = pd.DataFrame(data=subjects_rows)
recovery_df.to_csv(f"recovery_pop_df_{group_id}.csv", index=False)
