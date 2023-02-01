import numpy as np
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

BASE_PATH = Path('/storage/store2/work/bmalezie')
DATA_PATH = BASE_PATH / 'camcan-cdl'
SUBJECTS_PATH = [x for x in DATA_PATH.glob('**/*') if x.is_file()]

n_subjects = 50
subjects_path = np.random.choice(SUBJECTS_PATH, size=n_subjects, replace=False)


for this_subject_path in subjects_path:
    X = np.load(this_subject_path)
    X /= X.std()
