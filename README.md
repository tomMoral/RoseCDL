# RoseCDL

RObust and ScalablE Convolutional Dictionary Learning


# Installation

Before installing `rosecdl`, you should check that you have installed a version of `pytorch` that suits your hardware constraints. You can then install `rosecdl` from the source code by following these steps:

```bash
git clone https://github.com/tomMoral/RoseCDL.git
cd RoseCDL
pip install .
```

If you’d like to contribute to `rosecdl`, you should also install additional packages for code formatting, testing, and experiment dependencies. To do this, replace the `pip install` command above with:

```bash
pip install -e .[dev,experiments]
```
