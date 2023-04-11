import torch
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

from wfdb.io.record import rdrecord


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

    def __init__(self, db_dir='./apnea-ecg', group_id=None, window=10_000,
                 dtype=torch.float, device='cuda:1', seed=42):
        super().__init__()
        self.db_dir = Path(db_dir)
        self.group_id = group_id
        self.window = window
        self.dtype = dtype
        self.device = device
        self.seed = seed
        # get subjects
        subject_id_list = pd.read_csv(
            self.db_dir / "participants.tsv", sep='\t')['Record'].values
        if group_id is not None:
            self.subjects  = [id for id in subject_id_list if id[0] == group_id]
        else:
            self.subjects = subject_id_list
        self.n_subjects = len(self.subjects)
        # get signals' lengths
        random.seed(self.seed)
        random.shuffle(self.subjects)
        self.shapes_time = np.array([rdrecord(
            record_name=str(self.db_dir / subject_id)).sig_len - self.window
            for subject_id in self.subjects]).cumsum()
        
    def __len__(self):
        return self.shapes_time[-1]

    def __getitem__(self, idx):
        subject_idx = np.searchsorted(self.shapes_time, idx, side='right')
        ecg_record = rdrecord(
            record_name=str(self.db_dir / self.subjects[subject_idx]))
        time_idx = 0 if subject_idx == 0 else idx - self.shapes_time[subject_idx-1]
        X = ecg_record.p_signal[time_idx:time_idx+self.window, :].T

        return torch.tensor(X, dtype=self.dtype, device=self.device)
        # return X




def create_physionet_dataloader(db_dir, group_id=None, window=10_000,
                                dtype=torch.float, device='cuda:1',
                                mini_batch_size=10,
                                random_state=1234567890):

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
            generator=generator
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
    n_samples=None
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
                data,
                device=device,
                dtype=dtype,
                sto=sto,
                window=window,
                dimN=dimN
            ),
            batch_size=mini_batch_size,
            shuffle=True,
            generator=generator
        )
    elif isinstance(data, (str)):
        return torch.utils.data.DataLoader(
            MEGPopDataset(
                data,
                device=device,
                dtype=dtype,
                window=window,
                seed=random_state,
                n_samples=n_samples
            ),
            batch_size=mini_batch_size,
            shuffle=True,
            generator=generator
        )


class ConvSignalDataset(torch.utils.data.Dataset):
    """
    Dataset for Stochastic torch CDL
    Parameters
    ----------
    data: np.array
        Data to be processed
    window: int
        Size of minibatches window.
    """

    def __init__(self, data, device, dtype, sto,
                 window=None, dimN=1):
        super().__init__()
        self.sto = sto
        self.device = device
        self.dtype = dtype
        self.dimN = dimN
        if self.sto:
            self.data = data
            self.window = min(window, data.shape[1] - 1)
        else:
            self.data = torch.tensor(
                data,
                device=self.device,
                dtype=self.dtype
            )

    def __getitem__(self, idx):
        if self.sto and self.dimN == 1:
            return torch.tensor(
                self.data[:, idx * self.window: (idx+1) * self.window],
                device=self.device,
                dtype=self.dtype
            )
        elif self.sto and self.dimN == 2:
            idx_i = idx // self.window
            idx_j = idx % self.window
            return torch.tensor(
                self.data[
                    :,
                    idx_i * self.window: (idx_i+1) * self.window,
                    idx_j * self.window: (idx_j+1) * self.window
                ],
                device=self.device,
                dtype=self.dtype
            )
        else:
            return self.data

    def __len__(self):
        if self.sto:
            return self.data.shape[1] // self.window
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
                    index = idx - self.shapes_time[i-1]
                return torch.tensor(
                    data_norm[
                        :,
                        # index * self.window: (index + 1) * self.window
                        index:(index+self.window)
                    ],
                    dtype=self.dtype,
                    device=self.device
                )

    def __len__(self):
        return self.shapes_time[-1]


# import torch
# import os
# import random
# import numpy as np


# def create_conv_dataloader(
#     data,
#     device,
#     dtype,
#     mini_batch_size=10,
#     sto=False,
#     window=None,
#     random_state=2147483647,
#     dimN=1
# ):
#     """
#     Create dataset from ImageWoof

#     Parameters
#     ----------
#     data : np.array (n_channels, n_times)
#         Path to data
#     device : str
#         Device for computations
#     dtype : type
#         Type of tensors
#     mini_batch_size : int, optional
#         Size of mini batches, by default 10
#     random_state : int, optional
#         Seed, by default 2147483647

#     Returns
#     -------
#     torch.utils.data.DataLoader
#         Torch DataLoader
#     """
#     generator = torch.Generator()
#     generator.manual_seed(random_state)
#     return torch.utils.data.DataLoader(
#         ConvSignalDataset(
#             data,
#             device=device,
#             dtype=dtype,
#             sto=sto,
#             window=window,
#             dimN=dimN
#         ),
#         batch_size=mini_batch_size,
#         shuffle=True,
#         generator=generator
#     )


# class ConvSignalDataset(torch.utils.data.Dataset):
#     """
#     Dataset for Stochastic torch CDL
#     Parameters
#     ----------
#     data: np.array
#         Data to be processed
#     window: int
#         Size of minibatches window.
#     """

#     def __init__(self, data, device, dtype, sto,
#                  window=None, dimN=1):
#         super().__init__()
#         self.sto = sto
#         self.device = device
#         self.dtype = dtype
#         self.dimN = dimN
#         if self.sto:
#             self.data = data
#             self.window = min(window, data.shape[1] - 1)
#         else:
#             self.data = torch.tensor(
#                 data,
#                 device=self.device,
#                 dtype=self.dtype
#             )

#     def __getitem__(self, idx):
#         if self.sto and self.dimN == 1:
#             return torch.tensor(
#                 self.data[:, idx * self.window: (idx+1) * self.window],
#                 device=self.device,
#                 dtype=self.dtype
#             )
#         elif self.sto and self.dimN == 2:
#             idx_i = idx // self.window
#             idx_j = idx % self.window
#             return torch.tensor(
#                 self.data[
#                     :,
#                     idx_i * self.window: (idx_i+1) * self.window,
#                     idx_j * self.window: (idx_j+1) * self.window
#                 ],
#                 device=self.device,
#                 dtype=self.dtype
#             )
#         else:
#             return self.data

#     def __len__(self):
#         if self.sto:
#             return self.data.shape[1] // self.window
#         else:
#             return 1


# class MEGPopDataset(torch.utils.data.Dataset):
#     """
#     Dataset for stochastic CDL on MEG subjects

#     Parameters
#     ----------
#     path : str
#         Path to the data

#     Attributes
#     -------
#     path : str
#         Path to the data
#     """

#     def __init__(self, path, window, n_samples=None, seed=100):
#         super().__init__()
#         self.path = path
#         self.n_samples = n_samples
#         self.window = window
#         self.seed = seed
#         self.subjects, self.shapes_time = self.make_dataset()

#     def make_dataset(self):
#         """
#         Create list of subjects

#         Returns
#         -------
#         tuple (list, np.array)
#             List of paths for subjects to process
#             Cumulative sum of shapes for indexing
#         """
#         subjects = []
#         shapes_time = []
#         all_subjects = []
#         # Get all files in directory recursively
#         for root, _, files in os.walk(self.path):
#             for file in files:
#                 all_subjects.append(os.path.join(root, file))
#         random.seed(self.seed)
#         random.shuffle(all_subjects)
#         # Select given number of subjects and associated recording duration
#         for subject in all_subjects:
#             if self.n_samples is not None and len(subjects) >= self.n_samples:
#                 break
#             subjects.append(subject)
#             shapes_time.append(
#                 np.load(subject).shape[1] - self.window
#             )
#         return subjects, np.array(shapes_time).cumsum()

#     def __getitem__(self, idx):
#         print(idx)
#         for i in range(self.shapes_time.shape[0]):
#             if idx < self.shapes_time[i]:
#                 subject_path = self.subjects[i]
#                 data = np.load(subject_path)
#                 data_norm = data / data.std()
#                 if i == 0:
#                     index = idx
#                 else:
#                     index = idx - self.shapes_time[i-1]
#                 return data_norm[:, index:(index+self.window)]

#     def __len__(self):
#         return self.shapes_time[-1]

