import numpy as np
import pytest

from rosecdl.utils.utils_exp import evaluate_D_hat


@pytest.mark.parametrize("n_atoms", [5, 10, 20])
@pytest.mark.parametrize("n_atoms_true", [5, 10, 20])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("height", [20, 35])
@pytest.mark.parametrize("width", [20, 35])
@pytest.mark.parametrize("height_true", [20, 35])
@pytest.mark.parametrize("width_true", [20, 35])
def test_dictionary_recovery_2d(
    n_atoms, n_atoms_true, n_channels, height, width, height_true, width_true,
):
    if height_true > height or width_true > width:
        pytest.skip("True dimensions are greater than learned dimensions")
    if n_atoms_true > n_atoms:
        pytest.skip("Number of true atoms is greater than number of atoms")

    rng = np.random.RandomState(0)

    true_d = np.zeros((n_atoms_true, n_channels, height, width))
    true_d[:, :, :10, :10] = 1
    dict2 = rng.rand(n_atoms, n_channels, height, width)

    random_score = evaluate_D_hat(true_d, dict2)
    assert random_score <= 0.2

    exact_score = evaluate_D_hat(true_d, true_d)
    assert np.isclose(exact_score, 1.0)
