import numpy as np
import pytest
from wincdl.wincdl import WinCDL


@pytest.mark.parametrize(
    "n_samples,n_channels,n_components,kernel_size,support,epochs",
    [
        (16, 1, 5, 32, 100, 5),  # One channel
        (24, 3, 8, 48, 300, 5),  # Multiple channels
    ],
)
# TODO: Add "uv_constraint" to possible rank values
@pytest.mark.parametrize("rank", ["full"]) 
def test_positive_dictionary_1d(
    n_samples, n_channels, n_components, kernel_size, support, epochs, rank
):
    """Test positive dictionary constraint with 1D data"""
    # Setup 1D test data
    X = np.random.rand(n_samples, n_channels, support)

    # Initialize WinCDL with 1D configuration
    cdl = WinCDL(
        n_components=n_components,
        kernel_size=kernel_size,
        n_channels=n_channels,
        positive_D=True,
        epochs=epochs,
        lmbd=0.1,
        rank=rank,
    )

    cdl.fit(X)
    assert np.all(cdl.D_hat_ >= 0), "1D dictionary contains negative values"

@pytest.mark.parametrize(
    "n_samples,n_channels,n_components,kernel_size,height,width,epochs",
    [
        (16, 1, 5, (8, 8), 40, 40, 5),  # One channel
        (24, 3, 8, (8, 8), 64, 64, 5),  # Multiple channels
    ],
)
def test_positive_dictionary_2d(n_samples, n_channels, n_components, kernel_size, height, width, epochs):
    """Test positive dictionary constraint with 2D data"""
    # Setup 2D test data (batch, channel, height, width)
    X = np.random.rand(n_samples, n_channels, height, width)

    # Initialize WinCDL with 2D configuration
    cdl = WinCDL(
        n_components=n_components,
        kernel_size=kernel_size,
        n_channels=n_channels,
        positive_D=True,
        epochs=epochs,
        lmbd=0.1,
    )

    cdl.fit(X)
    assert np.all(cdl.D_hat_ >= 0), "2D dictionary contains negative values"
