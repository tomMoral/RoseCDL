"""
For a set a subjects, compute the recovery cost of one's dictionary on
another's signal, and save the matrix as a dataframe.
"""

# %%
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

from utils_apnea import get_subject_z_and_cost

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
    print(f"Get recovery DataFrame on group {group_id} ({group_des[group_id]})")
else:
    print("Get recovery DataFrame on all subjects")

subjects_rows = []
for subject_id in subject_id_list:
    subject_dir = Path(f"apnea-ecg/{subject_id}")
    with open(subject_dir / f"dict_res_{args.fit}.pkl", "rb") as f:
        dict_res = pickle.load(f)

    subjects_rows.extend(
        [
            dict(subject_id=subject_id, dict_fit="D_init", cost=dict_res["cost_init"]),
            dict(subject_id=subject_id, dict_fit=subject_id, cost=dict_res["cost"]),
        ]
    )

    # load random and final dictionaries
    subject_d_random = np.load(subject_dir / f"d_random_{args.fit}.npy")
    subject_d_hat = np.load(subject_dir / f"d_hat_{args.fit}.npy")

    # compute random dictionary cost
    _, cost = get_subject_z_and_cost(subject_id, subject_d_random)
    subjects_rows.append(dict(subject_id=subject_id, dict_fit="D_random", cost=cost))

    for other_subject_id in tqdm(subject_id_list):
        if subject_id == other_subject_id:
            continue
        _, cost = get_subject_z_and_cost(
            other_subject_id, subject_d_hat, label=args.fit
        )
        subjects_rows.append(
            dict(subject_id=subject_id, dict_fit=other_subject_id, cost=cost)
        )

recovery_df = pd.DataFrame(data=subjects_rows)
recovery_df.to_csv(f"recovery_df_{group_id}_{args.fit}.csv", index=False)
# %%
