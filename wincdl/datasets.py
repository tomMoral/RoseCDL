import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from wfdb.io.record import rdrecord


class PhysionetDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for loading Physionet ECG recordings.

    This dataset class is designed to load and process ECG recordings from the Physionet database,
    specifically for the apnea-ecg dataset. It provides functionality to load ECG signals
    for different subject groups and handles data preprocessing.

    Parameters
    ----------
    db_dir : str, default="./apnea-ecg"
        Path to the root directory containing the Physionet database files.

    group_id : str, optional
        Subject group identifier to filter the dataset:
        - 'a': apnea patients
        - 'b': borderline apnea patients
        - 'c': control subjects
        - 'x': test subjects
        If None, includes all groups.

    window : int, default=10000
        The size of the sliding window for ECG signal segmentation.

    dtype : torch.dtype, default=torch.float
        Data type for the output tensors.

    device : str, default="cuda:1"
        Device to store the tensors on (e.g., 'cpu', 'cuda:0', 'cuda:1').

    seed : int, default=42
        Random seed for reproducibility when shuffling subjects.

    Attributes
    ----------
    subjects : list
        List of subject IDs included in the dataset.
    n_subjects : int
        Number of subjects in the dataset.
    shapes_time : numpy.ndarray
        Cumulative sum of signal lengths for all subjects.

    Returns
    -------
    torch.Tensor
        ECG signal segment of shape (channels, window) normalized by standard deviation.
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
        random.seed(self.seed)
        random.shuffle(self.subjects)
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
        # return X


def create_physionet_dataloader(
    db_dir,
    group_id=None,
    window=10_000,
    dtype=torch.float,
    device="cuda:1",
    mini_batch_size=10,
    random_state=1234567890,
):
    generator = torch.Generator()
    generator.manual_seed(random_state)

    return torch.utils.data.DataLoader(
        PhysionetDataset(
            db_dir=db_dir,
            group_id=group_id,
            window=window,
            dtype=dtype,
            device=device,
            seed=random_state,
        ),
        batch_size=mini_batch_size,
        shuffle=True,
        generator=generator,
    )


def create_conv_dataloader(
    data,
    device,
    dtype,
    mini_batch_size=10,
    sto=False,
    window=None,
    random_state=2147483647,
    dimN=1,
    n_samples=None,
):
    """
    Create dataset for conv signals

    Parameters
    ----------
    data : str or np.array (n_channels, n_times)
        Path to data or np.array
    device : str
        Device for computations
    dtype : type
        Type of tensors
    mini_batch_size : int, optional
        Size of mini batches, by default 10
    random_state : int, optional
        Seed, by default 2147483647

    Returns
    -------
    torch.utils.data.DataLoader
        Torch DataLoader
    """
    generator = torch.Generator()
    generator.manual_seed(random_state)
    if isinstance(data, np.ndarray):
        return torch.utils.data.DataLoader(
            ConvSignalDataset(
                data, device=device, dtype=dtype, sto=sto, window=window, dimN=dimN
            ),
            batch_size=mini_batch_size,
            shuffle=True,
            generator=generator,
        )
    elif isinstance(data, (str)):
        return torch.utils.data.DataLoader(
            MEGPopDataset(
                data,
                device=device,
                dtype=dtype,
                window=window,
                seed=random_state,
                n_samples=n_samples,
            ),
            batch_size=mini_batch_size,
            shuffle=True,
            generator=generator,
        )


class ConvSignalDataset(torch.utils.data.Dataset):
    """Dataset for convolutional signal processing using PyTorch.

    A dataset class that handles both stochastic and deterministic processing of signal data,
    supporting both 1D and 2D convolution operations.

    data : numpy.ndarray
        Input signal data. For 1D convolution shape should be (channels, length).
        For 2D convolution shape should be (channels, height, width).
    device : torch.device
        Device to store the tensors on (CPU/GPU).
    dtype : torch.dtype
        Data type for the tensors (e.g., torch.float32).
    sto : bool
        If True, enables stochastic processing by windowing the data.
        If False, processes the entire signal at once.
    window : int, optional
        Size of the sliding window for stochastic processing.
        If None, uses full signal length. Default is None.
    dimN : int, optional
        Dimensionality of the convolution operation.
        1 for 1D convolution, 2 for 2D convolution. Default is 1.

    Attributes
    data : Union[numpy.ndarray, torch.Tensor]
        The stored signal data, either as numpy array (stochastic)
        or torch tensor (deterministic).
    window : int
        The effective window size used for processing.

    Methods
    -------
    __getitem__(idx)
        Returns a windowed segment of the data for stochastic processing,
        or the full data for deterministic processing.
    __len__()
        Returns the number of available windows for stochastic processing,
        or 1 for deterministic processing.
    """

    def __init__(self, data, device, dtype, sto, window=None, dimN=1):
        super().__init__()
        self.sto = sto
        self.device = device
        self.dtype = dtype
        self.dimN = dimN
        if self.sto:
            self.data = data
            self.window = min(window, data.shape[1])
            # self.window = min(window, data.shape[1] - 1)
        else:
            self.data = torch.tensor(data, device=self.device, dtype=self.dtype)

    def __getitem__(self, idx):
        if self.sto and self.dimN == 1:
            return torch.tensor(
                self.data[:, idx : (idx + self.window)],
                # self.data[:, idx * self.window: (idx+1) * self.window],
                device=self.device,
                dtype=self.dtype,
            )
        elif self.sto and self.dimN == 2:
            idx_i = idx // self.window
            idx_j = idx % self.window
            return torch.tensor(
                self.data[
                    :,
                    idx_i * self.window : (idx_i + 1) * self.window,
                    idx_j * self.window : (idx_j + 1) * self.window,
                ],
                device=self.device,
                dtype=self.dtype,
            )
        else:
            return self.data

    def __len__(self):
        if self.sto:
            return self.data.shape[1] - self.window + 1
            # return self.data.shape[1] // self.window
        else:
            return 1


class MEGPopDataset(torch.utils.data.Dataset):
    """
    Dataset for stochastic CDL on MEG subjects

    Parameters
    ----------
    path : str
        Path to the data

    Attributes
    -------
    path : str
        Path to the data
    """

    def __init__(self, path, window, dtype, device, seed, n_samples=None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.path = path
        self.n_samples = n_samples
        self.window = window
        self.seed = seed
        self.subjects, self.shapes_time = self.make_dataset()

    def make_dataset(self):
        """
        Create list of subjects

        Returns
        -------
        tuple (list, np.array)
            List of paths for subjects to process
            Cumulative sum of shapes for indexing
        """
        subjects = []
        shapes_time = []
        all_subjects = []
        # Get all files in directory recursively
        for root, _, files in os.walk(self.path):
            for file in files:
                all_subjects.append(os.path.join(root, file))
        random.seed(self.seed)
        random.shuffle(all_subjects)
        # Select given number of subjects and associated recording duration
        for subject in all_subjects:
            if self.n_samples is not None and len(subjects) >= self.n_samples:
                break
            subjects.append(subject)
            shapes_time.append(
                # np.load(subject).shape[1] // self.window
                np.load(subject).shape[1] - self.window
            )
        return subjects, np.array(shapes_time).cumsum()

    def __getitem__(self, idx):
        print(idx)
        for i in range(self.shapes_time.shape[0]):
            if idx < self.shapes_time[i]:
                print(f"i = {i}")
                subject_path = self.subjects[i]
                data = np.load(subject_path)
                data_norm = data / data.std()
                if i == 0:
                    index = idx
                else:
                    index = idx - self.shapes_time[i - 1]
                return torch.tensor(
                    data_norm[
                        :,
                        # index * self.window: (index + 1) * self.window
                        index : (index + self.window),
                    ],
                    dtype=self.dtype,
                    device=self.device,
                )

    def __len__(self):
        return self.shapes_time[-1]
