import numpy as np

from rosecdl.utils.dictionary import tukey_window_2d


def test_tukey_window_2d():
    n_rows = 3
    n_cols = 4

    expected_result = np.ones((n_rows, n_cols))
    expected_result[0, :] = 1e-9
    expected_result[-1, :] = 1e-9
    expected_result[:, 0] = 1e-9
    expected_result[:, -1] = 1e-9

    assert np.allclose(tukey_window_2d(n_rows, n_cols), expected_result)
