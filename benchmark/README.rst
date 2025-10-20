
RoseCDL Benchmark
=================
|Build Status| |Python 3.10+|

This benchmark evaluates the runtime and scaling of RoseCDL for Convolutional
Dictionary Learning (CDL). This problem aims to learn a dictionary of atoms
that can be used to represent a signal $X$ as a convolution between a dictionary
of patterns denoted $D$ which contains prototypical patches of the signal, and
some sparse codes $z_k$ that indicate where these atoms are located in the
signal's domain. The signal's model reads:

$$X = \\sum_{k=1}^K z_k * D_k$$

To infer the parameters $D$, we can use various optimization methods.
RoseCDL has been proposed in `Yehya et al., 2025 <https://arxiv.org/abs/2509.07523>`_.
We evaluate here its runtime and scaling compared to other methods such as:

- `Sporco <https://github.com/bwohlberg/sporco>`_
- `AlphaCSC <https://github.com/alphacsc/alphacsc>`_

To evaluate if a method is good, we can compute several scores:
- The loss on a left out part of the data,
- The recovery score when D is known.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tommoral/RoseCDL
   $ benchopt run RoseCDL/benchmark

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run RoseCDL/benchmark --config configs/test.yml


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tommoral/RoseCDL/workflows/Benchopt/badge.svg
   :target: https://github.com/tommoral/RoseCDL/actions
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
