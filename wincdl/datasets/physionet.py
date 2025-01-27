from pathlib import Path

import torch
import numpy as np

try:
    import wfdb
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
        sample_window=10_000,
        download=False,
        device="cuda:1",
        dtype=torch.float,
    ):
        super().__init__()
        self.db_dir = Path(db_dir)
        self.group_id = group_id
        self.sample_window = sample_window
        self.dtype = dtype
        self.device = device

        if not self.db_dir.exists():
            if not download:
                raise FileNotFoundError("The Physionet data was not found")
            print("Downloading data...", end='', flush=True)
            wfdb.dl_database("apnea-ecg", dl_dir=str(self.db_dir))
            print("ok")

        # get subjects
        subject_id_list = sorted(list(set(
            f.with_suffix('').name for f in self.db_dir.glob("*")
        )))
        subject_id_list = [f for f in subject_id_list if 'r' not in f]

        if group_id is not None:
            self.subjects = [id for id in subject_id_list if id[0] == group_id]
        else:
            self.subjects = subject_id_list
        self.n_subjects = len(self.subjects)
        # get signals' lengths
        self._shape_window = np.array([
            wfdb.rdrecord(record_name=str(self.db_dir / subject_id)).sig_len
            for subject_id in self.subjects
        ])
        self.shapes_time = np.cumsum(
            np.maximum(1, self._shape_window - self.sample_window)
        )
        self.n_windows = int(sum(
            np.maximum(1, T // self.sample_window) for T in self._shape_window
        ))

    def __len__(self):
        return self.shapes_time[-1]

    def __getitem__(self, idx):
        subject_idx = np.searchsorted(self.shapes_time, idx, side="right")
        time_idx = 0 if subject_idx == 0 else idx - self.shapes_time[subject_idx - 1]
        ecg_record = wfdb.rdrecord(
            record_name=str(self.db_dir / self.subjects[subject_idx]),
            sampfrom=time_idx,
            sampto=time_idx+self.sample_window,
        )
        X = ecg_record.p_signal.T
        X /= X.std()

        return torch.tensor(X, dtype=self.dtype, device=self.device)
