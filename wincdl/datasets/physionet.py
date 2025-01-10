from pathlib import Path

import torch
import numpy as np
import pandas as pd

try:
    from wfdb.io.record import rdrecord
except ImportError:
    raise ImportError(
        "The physionet dataset requires wfdb to be installed. Please run "
        "`pip install wfdb` to use this class."
    )


class PhysionetDataset(torch.utils.data.Dataset):
    """

    db_dir : string
        path to the dataset
        values can be 'apnea-ecg'

    group_id : string
        group label to create the dataset on
        'a': apnea, 'b': borderline apnea, 'c': control, 'x': test
        if None, all
        default is None

    seed : int
        random seed

    """

    def __init__(
        self,
        db_dir="./apnea-ecg",
        group_id=None,
        window=10_000,
        dtype=torch.float,
        device="cuda:1",
        seed=42,
    ):
        super().__init__()
        self.db_dir = Path(db_dir)
        self.group_id = group_id
        self.window = window
        self.dtype = dtype
        self.device = device
        self.seed = seed
        # get subjects
        subject_id_list = pd.read_csv(self.db_dir / "participants.tsv", sep="\t")[
            "Record"
        ].values
        if group_id is not None:
            self.subjects = [id for id in subject_id_list if id[0] == group_id]
        else:
            self.subjects = subject_id_list
        self.n_subjects = len(self.subjects)
        # get signals' lengths
        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.subjects)
        self.shapes_time = np.array(
            [
                rdrecord(record_name=str(self.db_dir / subject_id)).sig_len
                - self.window
                for subject_id in self.subjects
            ]
        ).cumsum()

    def __len__(self):
        return self.shapes_time[-1]

    def __getitem__(self, idx):
        subject_idx = np.searchsorted(self.shapes_time, idx, side="right")
        ecg_record = rdrecord(record_name=str(self.db_dir / self.subjects[subject_idx]))
        time_idx = 0 if subject_idx == 0 else idx - self.shapes_time[subject_idx - 1]
        X = ecg_record.p_signal
        X /= X.std()
        X = X[time_idx : time_idx + self.window, :].T

        return torch.tensor(X, dtype=self.dtype, device=self.device)
