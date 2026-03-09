<div align="center">

# RoseCDL: Robust and Scalable Convolutional Dictionary Learning for rare-event and anomaly detection

[![arXiv](https://img.shields.io/static/v1?label=ArXiv&message=2509.07523&color=B31B1B&logo=arXiv)](https://arxiv.org/pdf/2509.07523)
[![Conference](https://img.shields.io/badge/AISTATS-2026-4b44ce.svg)](https://aistats.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/mit)

</div>

RoseCDL is a convolutional dictionary learning method that learns patterns from data in an unsupervised manner. It includes an Inline Outlier detection module that allows it to be SOTA in Rare event and anomaly detection tasks. Additionaly, it is highly scalable and 2 orders of magnitude faster than other methods.

# Installation

> [!TIP]
> Before installing `rosecdl`, you should check that you have installed a version of `pytorch` that suits your hardware constraints.
You can then install `rosecdl` from the source code by following these steps:

```bash
git clone https://github.com/tomMoral/RoseCDL.git
cd RoseCDL
pip install .
```

# Example

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

# Contributing

If you’d like to contribute to `rosecdl`, you should also install additional packages for code formatting, testing, and experiment dependencies. To do this, replace the `pip install` command above with:

```bash
pip install -e .[dev,experiments]
```
