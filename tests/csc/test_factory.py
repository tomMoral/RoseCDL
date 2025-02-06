import pytest

from rosecdl.csc.factory import csc_factory


@pytest.mark.parametrize(
    "kernel_size, rank1, expected_cls",
    [
        ((10,), False, "CSC1d"),
        ((10,), True, "Rank1CSC1d"),
        ((10, 10), False, "CSC2d"),
    ],
)
def test_csc_factory(kernel_size, rank1, expected_cls):

    base_dict = {
        "lmbd": 0.1,
        "n_components": 4,
        "n_channels": 2,
        "D_init": None,
        "window": False,
        "positive_D": True,
        "positive_z": True,
        "n_iterations": 20,
        "deepcdl": False,
        "random_state": None,
        "device": None,
        "dtype": float,
    }

    kwargs = dict(**base_dict, kernel_size=kernel_size, rank1=rank1)
    assert type(csc_factory(**kwargs)).__name__ == expected_cls
