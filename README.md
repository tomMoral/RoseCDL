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

TBD

# Contributing

If you’d like to contribute to `rosecdl`, you should also install additional packages for code formatting, testing, and experiment dependencies. To do this, replace the `pip install` command above with:

```bash
pip install -e .[dev,experiments]
```

## Citation

If you use RoseCDL in your research, please cite:

TBD
