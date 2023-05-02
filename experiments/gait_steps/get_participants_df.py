import pandas as pd
import json
from tqdm import tqdm
from dicodile.data.gait import get_gait_data

CODE_LIST_FNAME = "code_list.json"

def get_code_list():
    """Returns the list of all available codes.
    Returns
    -------
    list
        List of codes.
    """
    with open(CODE_LIST_FNAME, "r") as f:
        code_list = json.load(f)
    return code_list


def get_subject_trial(subject_id):

    all_codes = get_code_list()
    n_subjects = int(all_codes[-1].split('-')[0])

    assert subject_id <= n_subjects, f"subject_id must be <= {n_subjects}"

    subject_trials = [code for code in all_codes 
                      if code.split('-')[0]==str(subject_id)]

    return subject_trials


all_codes = get_code_list()
n_subjects = int(all_codes[-1].split('-')[0])

subject_rows = []
for subject_id in tqdm(range(1, n_subjects+1)):
    n_trials = len(get_subject_trial(subject_id))
    trial = get_gait_data(subject=subject_id, trial=1)

    subject_rows.append(dict(
        Subject=subject_id,
        n_trials=n_trials,
        age=trial['Age'],
        Gender=trial['Gender'],
        Height=trial['Height'],
        Weight=trial['Weight'],
        BMI=trial['BMI'],
        Laterality=trial['Laterality'],
        Sensor=trial['Sensor'],
        WalkedDistance=trial['WalkedDistance'],
        WalkingSpeed=trial['WalkingSpeed'],
        PathologyGroup=trial['PathologyGroup'],
        IsControl=trial['IsControl'],

    ))

participants = pd.DataFrame(subject_rows)
participants.to_csv('particants.tsv', sep='\t')