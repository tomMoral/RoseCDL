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

TBD

# Contributing

If you’d like to contribute to `rosecdl`, you should also install additional packages for code formatting, testing, and experiment dependencies. To do this, replace the `pip install` command above with:

```bash
pip install -e .[dev,experiments]
```
