import numpy as np
import pytest
import torch

from rosecdl.rosecdl import RoseCDL


@pytest.mark.parametrize(
    "kernel_size, support",
    [
        (32, (100,)),  # 1D
        ((8, 9), (40, 64)),  # 2D
    ],
)
# TODO: Add True to possible rank1 values
@pytest.mark.parametrize("rank1", [False])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("n_components", [3, 5])
@pytest.mark.parametrize("positive_D", [True, False])
def test_rosecdl(n_components, n_channels, kernel_size, support, rank1, positive_D):
    """Test positive dictionary constraint with 1D and 2D data"""
    # Setup test data
    X = torch.rand(10, n_channels, *support)

    # Initialize RoseCDL and fit it
    cdl = RoseCDL(
        n_components=n_components,
        kernel_size=kernel_size,
        n_channels=n_channels,
        positive_D=positive_D,
        epochs=1,
        lmbd=0.1,
        rank1=rank1,
    )
    cdl.fit(X)

    # Assert the shape of the dictionary
    kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    assert cdl.D_hat_.shape == (n_components, n_channels, *kernel_size), (
        "Dictionary shape is not correct. Expected: "
        f"{(n_components, n_channels, *kernel_size)}, got: {cdl.D_hat_.shape}"
    )

    # If positive_D is True, assert that all values in the dictionary are positive
    if positive_D:
        assert np.all(cdl.D_hat_ >= 0), (
            f"{len(support)}D dictionary contains negative values, "
            f"while using positive_D=True"
        )
