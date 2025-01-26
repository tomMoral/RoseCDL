
CDL Benchmark
=============
|Build Status| |Python 3.10+|

This benchmark is dedicated to Convolutional Dictionary learning methods.
These methods aim to learn a dictionary of atoms that can be used to represent
a signal $X$ as a convolution between a dictionary of patterns denoted $D$ which
contains prototypical patches of the signal, and some sparse codes $z_k$ that
indicate where these atoms are located in the signal's domain.
The signal's model reads:


$$X = \sum_{k=1}^K z_k * D_k$$

To evaluate if a method is good, we can compute several scores:
- The test loss
- The recovery score when D is known.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tommoral/WinCDL
   $ benchopt run WinCDL/benchmark

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run WinCDL/benchmark -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tommoral/WinCDL/workflows/Benchopt/badge.svg
   :target: https://github.com/tommoral/WinCDL/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
