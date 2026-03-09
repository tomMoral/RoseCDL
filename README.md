<div align="center">

# RoseCDL: Robust and Scalable Convolutional Dictionary Learning for rare-event and anomaly detection

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2509.07523&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2509.07523)
[![Conference](https://img.shields.io/badge/AISTATS-2026-4b44ce.svg)](https://aistats.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)

</div>

RoseCDL learns convolutional dictionaries from raw signals and images in a fully unsupervised manner, while **simultaneously** detecting anomalies and rare events. Unlike prior approaches that require a separate pre- or post-processing step for outlier handling, RoseCDL integrates a robust trimmed-loss mechanism directly into the dictionary learning loop, yielding cleaner dictionaries **and** accurate anomaly masks in a single pass.


## Installation

> [!TIP]
> Before installing `rosecdl`, ensure you have a version of [PyTorch](https://pytorch.org/get-started/locally/) that matches your hardware (CPU / CUDA).

```bash
git clone https://github.com/tomMoral/RoseCDL.git
cd RoseCDL
pip install .
```

## Quick Start

### Dictionary Learning

```python
from rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat
from rosecdl.utils.utils_signal import generate_experiment

simulation_params = {
    "n_trials": 10,
    "n_times": 5_000,
    "n_atoms": 2,
    "n_times_atom": 128,
    "window": True,
    "contamination_params": None,
}

X, _, D_true, D_init = generate_experiment(simulation_params)

cdl = RoseCDL(lmbd=0.8, D_init=D_init, epochs=30, sample_window=1000)
cdl.fit(X)

recovery = evaluate_D_hat(D_true, cdl.D_hat_)
print(f"Dictionary recovery score: {recovery:.4f}")
```

### Anomaly Detection

```python
import numpy as np
from rosecdl import RoseCDL
from rosecdl.utils.utils_signal import generate_experiment
from sklearn.metrics import f1_score

# Generate 1D signal with injected anomalies
simulation_params = {
    "n_trials": 10,
    "n_channels": 1,
    "n_times": 5_000,
    "n_atoms": 2,
    "n_times_atom": 64,
    "n_atoms_extra": 2,
    "D_init": "random",
    "window": True,
    "init_d": "shapes",
    "init_d_kwargs": {"shapes": ["sin", "gaussian"]},
    "init_z": "constant",
    "init_z_kwargs": {"value": 1},
    "noise_std": 0.01,
    "sparsity": 20,
    "n_patterns_per_atom": 1,
    "contamination_params": {
        "n_atoms": 2,
        "sparsity": 3,
        "init_z": "constant",
        "init_z_kwargs": {"value": 50},
    },
    "rng": 42,
}

X, z, D_true, D_init, info = generate_experiment(
    simulation_params, return_info_contam=True
)

# Fit with inline outlier detection (MAD method, alpha=3.5)
cdl = RoseCDL(
    lmbd=0.8,
    scale_lmbd=True,
    D_init=D_init,
    epochs=30,
    n_iterations=50,
    sample_window=960,
    outliers_kwargs={"method": "mad", "alpha": 3.5},
)
cdl.fit(X)

# Retrieve anomaly mask
pred_mask = cdl.get_outlier_mask(X)

# Evaluate
true_mask = info["outliers_mask"].max(axis=1, keepdims=True)
f1 = f1_score(true_mask.flatten(), pred_mask.flatten())
print(f"Anomaly detection F1 score: {f1:.4f}")
```

## Contributing

```bash
pip install -e .[dev,experiments]
```

## Citation

If you use RoseCDL in your research, please cite:

TBD
