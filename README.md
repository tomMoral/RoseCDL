<div align="center">

# RoseCDL: Robust and Scalable Convolutional Dictionary Learning for rare-event and anomaly detection

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2509.07523&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2509.07523)
[![Conference](https://img.shields.io/badge/AISTATS-2026-4b44ce.svg)](https://aistats.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

</div>

## Abstract

> Identifying recurring patterns and rare events in large-scale signals is a fundamental challenge in fields such as astronomy, physical simulations, and biomedical science. Convolutional Dictionary Learning (CDL) offers a powerful framework for modeling local structures in signals, but its use for detecting rare or anomalous events remains largely unexplored. In particular, CDL faces two key challenges in this setting: high computational cost and sensitivity to artifacts and outliers. In this paper, we introduce RoseCDL, a scalable and robust CDL algorithm designed for unsupervised rare event detection in long signals. RoseCDL combines stochastic windowing for efficient training on large datasets with inline outlier detection to enhance robustness and isolate anomalous patterns. This reframes CDL as a practical tool for event discovery and characterization in real-world signals, extending its role beyond traditional tasks like compression or denoising.


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
from rosecdl.rosecdl import RoseCDL
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

# Generating simulated data
data, _, true_dict, _ = generate_experiment(simulation_params)

rosecdl = RoseCDL(
    n_components=3,
    kernel_size=128,
    n_channels=1,
    lmbd=0.8,
    n_iterations=30,
    epochs=30,
    sample_window=1000,
)

# Fitting RoseCDL on data
rosecdl.fit(data)
learned_dict = rosecdl.D_hat_

# Computing the recovery score of the learned dictionary
recovery_score = evaluate_D_hat(learned_dict, true_dict)
print("Dictionary recovery score : ", recovery_score)
```

### Anomaly Detection

```python
import numpy as np
from rosecdl import RoseCDL
from rosecdl.utils.utils_signal import generate_experiment
from rosecdl.utils.utils_exp import get_outliers_metric
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

X, _, true_dict, _, info = generate_experiment(
    simulation_params, return_info_contam=True
)

# Fit with inline outlier detection (MAD method, alpha=3.5)
cdl = RoseCDL(
    n_components=4,
    kernel_size=64,
    n_channels=1,
    lmbd=0.8,
    n_iterations=30,
    epochs=30,
    sample_window=960,
    outliers_kwargs={"method": "mad", "alpha": 3.5},
)
cdl.fit(X)

# Evaluate
true_mask = info["outliers_mask"].max(axis=1, keepdims=True)
metrics = get_outliers_metric(
    true_mask, cdl, X, crop=True
)

print("\nAnomaly detection metrics:")
for name, score in metrics.items():
    print(f"{name:12}: {score:.4f}")
```

# Contributing

If you’d like to contribute to `rosecdl`, you should also install additional packages for code formatting, testing, and experiment dependencies. To do this, replace the `pip install` command above with:

```bash
pip install -e .[dev,experiments]
```

## Citation

If you use RoseCDL in your research, please cite:

```
@inproceedings{
yehya2026rosecdl,
title={Rose{CDL}: Robust and Scalable Convolutional Dictionary Learning for rare-event and anomaly detection},
author={Jad Yehya and Mansour Benbakoura and C{\'e}dric Allain and Beno{\^\i}t Mal{\'e}zieux and Matthieu Kowalski and Thomas Moreau},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=4XMkOFxxfb}
}
```
