import numpy as np
import pytest

from rosecdl.utils.utils_exp import evaluate_D_hat


@pytest.mark.parametrize("n_atoms", [5, 10])
@pytest.mark.parametrize("extra_n_atoms", [0, 5])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("height", [20, 35])
@pytest.mark.parametrize("width", [20, 35])
@pytest.mark.parametrize("size_increase", [0, 5])
def test_dictionary_recovery_2d(
    n_atoms, extra_n_atoms, n_channels, height, width, size_increase
):
    # One seed per config to avoid having too correlated tests
    seed = hash((n_atoms, extra_n_atoms, n_channels, height, width, size_increase))
    rng = np.random.RandomState(seed % (2**32 - 1))

    true_d = rng.rand(n_atoms, n_channels, height, width)
    true_d[:, :, :5, :5] = 3
    dict2 = rng.rand(
        n_atoms + extra_n_atoms,
        n_channels,
        height + size_increase,
        width + size_increase,
    )

    random_score = evaluate_D_hat(true_d, dict2)
    assert random_score <= 0.2

    exact_score = evaluate_D_hat(true_d, true_d)
    assert np.isclose(exact_score, 1.0)

    true_d_extra = dict2.copy()
    true_d_extra[:n_atoms, :, :height, :width] = true_d
    exact_score = evaluate_D_hat(true_d, true_d_extra)
    assert np.isclose(exact_score, 1.0)
