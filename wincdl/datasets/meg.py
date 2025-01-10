import os

import torch
import numpy as np



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
        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_subjects)
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
        for i in range(self.shapes_time.shape[0]):
            if idx < self.shapes_time[i]:
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
