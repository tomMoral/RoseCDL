from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sawtooth


class Wave:
    """Base class for generating waves.

    Subclasses should implement the `generate` method.

    Parameters
    ----------
    n_points : int
        The number of points in the wave.
    frequency : int
        The frequency of the wave. This is the number of cycles per wave.
    positive_only : bool
        If True, the wave will be positive only. This is useful for generating half-waves.

    Attributes
    ----------
    n_points : int
        The number of points in the wave.
    frequency : int
        The frequency of the wave. This is the number of cycles per wave.
    positive_only : bool
        If True, the wave will be positive only. This is useful for generating half-waves.

    Methods
    -------
    generate()
        Generate the wave.

    """  # noqa: E501

    def __init__(self, n_points, frequency=1, positive_only=False):
        self.n_points = n_points
        self.frequency = frequency
        self.positive_only = positive_only

    def generate(self):
        pass  # This method should be overridden by subclasses

    def plot(self):
        """Plot the wave."""
        wave = self.generate()
        plt.plot(wave)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(f"{self.__class__.__name__} wave at frequency {self.frequency}")
        plt.show()


class SinWave(Wave):
    """Sine wave."""

    def generate(self):
        # Generate a sinusoidal wave of period `n_points / frequency`
        x = np.linspace(0, 2 * np.pi * self.frequency, self.n_points)
        wave = np.sin(x)
        if self.positive_only:
            # Ensure that the wave is positive, resulting in a half-wave
            wave = np.abs(wave)
        return wave


class SquareWave(Wave):
    """Square wave."""

    def generate(self):
        # Generate a square wave of period `n_points / frequency`
        wave = np.sign(
            np.sin(
                2 * np.pi * self.frequency * np.arange(self.n_points) / self.n_points
            )
        )
        wave[-1] = 0  # ensure it ends at 0
        if self.positive_only:
            # Ensure that the wave is positive, resulting in a pulse wave
            wave = np.maximum(wave, 0)
        return wave


class SawtoothWave(Wave):
    """Sawtooth wave."""

    def generate(self):
        # Generate a sawtooth wave of period `n_points / frequency`
        wave = (
            sawtooth(
                2 * np.pi * self.frequency * np.arange(self.n_points) / self.n_points
            )
            + 1
        ) / 2  # goes from 0 to 1
        wave[-1] = 0  # ensure it ends at 0
        if self.positive_only:
            wave = np.abs(wave)
        return wave


class GaussianWave(Wave):
    """Gaussian wave."""

    def generate(self):
        # Generate a Gaussian wave. This wave is produced by summing multiple Gaussian
        # functions with their means spaced equally across the wave.
        # The number of Gaussians summed is equal to the frequency.
        gaussians = []
        variance = self.n_points / self.frequency
        for i in range(self.frequency):
            mean = ((i + 1) / (self.frequency + 1)) * self.n_points
            x = np.linspace(0, self.n_points - 1, self.n_points)
            gaussian = np.exp(-np.power(x - mean, 2.0) / (2 * variance))
            gaussians.append(gaussian)
        wave = np.sum(gaussians, axis=0)
        if self.positive_only:
            wave = np.abs(wave)
        return wave


class TriangleWave(Wave):
    """Triangle wave."""

    def generate(self):
        # Generate a triangle wave of period `n_points / frequency`
        t = np.linspace(0, 1, self.n_points)
        wave = (
            2 * np.abs(2 * (t * self.frequency - np.floor(0.5 + t * self.frequency)))
            - 1
        )
        if self.positive_only:
            # Ensure that the wave is positive, resulting in a wave that rises from 0,
            # peaks to 1, and falls back to 0.
            wave = np.maximum(wave, 0)
        return wave


class WaveFactory:
    """A factory class for generating wave objects of various types.

    This class provides a method to generate wave objects of various types. The types of waves that can be generated
    are defined in the wave_classes attribute, which is a dictionary mapping from string names to wave classes.

    The factory maintains a cycle of the wave types, and a method to get the next wave in the cycle. It also maintains
    a dictionary of current frequencies for each wave type, which is incremented each time a wave of that type is generated.

    Parameters
    ----------
    start_freq : int, optional (default=1)
        The starting frequency for each wave type.
    shapes : list of str, optional
        The list of wave shapes to cycle through. If not provided, all available shapes are used.

    Attributes
    ----------
    wave_classes : dict
        A dictionary mapping from string names to wave classes.
    wave_cycle : cycle
        A cycle of the wave types, as defined in wave_classes or the provided shapes list.
    current_frequencies : dict
        A dictionary mapping from wave types to the current frequency for that type.

    Methods
    -------
    create_wave(wave_type, n_points, frequency=1, positive_only=False)
        Create a wave of the specified type.
    next_wave(n_points, frequency=None, positive_only=False)
        Create a wave of the next type in the cycle, with an incrementing frequency.

    """  # noqa: E501

    def __init__(self, start_freq=1, shapes=None):
        self.wave_classes = {
            "sin": SinWave,
            "square": SquareWave,
            "sawtooth": SawtoothWave,
            "triangle": TriangleWave,
            "gaussian": GaussianWave,
        }
        if shapes is None:  # If no shapes are specified, use all available shapes
            shapes = list(self.wave_classes.keys())
        else:  # Check that the provided shapes are valid
            for shape in shapes:
                if shape not in self.wave_classes:
                    raise ValueError(
                        f"Invalid shape: '{shape}'. "
                        f"Valid shapes are: {list(self.wave_classes.keys())}"
                    )
        self.wave_cycle = cycle(shapes)
        self.current_frequencies = dict.fromkeys(shapes, start_freq)

    def create_wave(self, wave_type, n_points, frequency=1, positive_only=False):
        """Create a wave of the specified type.

        Parameters
        ----------
        wave_type : str
            The type of wave to create. Must be a key in self.wave_classes.
        n_points : int
            The number of points in the wave.
        frequency : int, optional (default=1)
            The frequency of the wave.
        positive_only : bool, optional (default=False)
            If True, the function will ensure that only positive values are generated.

        Returns
        -------
        wave : Wave
            The generated wave object.

        """
        if wave_type not in self.wave_classes:
            raise ValueError(
                f"Invalid wave_type: '{wave_type}'. "
                f"Valid types are: {list(self.wave_classes.keys())}"
            )
        return self.wave_classes[wave_type](n_points, frequency, positive_only)

    def next_wave(self, n_points, frequency=None, positive_only=False):
        """Create a wave of the next type in the cycle, with an incrementing frequency.

        Parameters
        ----------
        n_points : int
            The number of points in the wave.
        frequency : int, optional
            The frequency of the wave. If not provided, the function will use and then increment the current frequency for the wave type.
        positive_only : bool, optional (default=False)
            If True, the function will ensure that only positive values are generated.

        Returns
        -------
        wave : Wave
            The generated wave object.

        """  # noqa: E501
        wave_type = next(self.wave_cycle)
        if (
            frequency is None
        ):  # If frequency is not specified, use the current frequency for the wave type
            frequency = self.current_frequencies[wave_type]
            self.current_frequencies[wave_type] += (
                1  # Increment the frequency for next time
            )
        return self.create_wave(wave_type, n_points, frequency, positive_only)


def plot_shapes(shapes=None, max_frequency=5, n_points=500, positive_only=True):
    """Generate and plot a set of waves with varying shapes and frequencies.

    This function generates and plots waves for a given set of shapes and frequencies up to the max_frequency.
    The purpose is to visualize the different shapes and frequencies that can be generated.

    Parameters
    ----------
    shapes : list of str, optional (default=['sin', 'gaussian'])
        The names of the wave classes to use.
        Available options are: 'sin', 'square', 'sawtooth', 'triangle', 'gaussian'
    max_frequency : int
        The maximum frequency to use for the wave generation functions.
    n_points : int, optinal (default=500)
        The number of points in the wave.
    positive_only : bool, optional (default=True)
        If True, the function will ensure that only positive values are generated.

    """  # noqa: E501
    if shapes is None:
        shapes = ["sin", "gaussian"]
    # Validate inputs
    factory = WaveFactory()
    if max_frequency < 1:
        raise ValueError("max_frequency must be at least 1")
    if not isinstance(positive_only, bool):
        raise TypeError("positive_only must be a boolean value")

    # Generate and plot waves

    fig, axs = plt.subplots(
        max_frequency,
        len(shapes),
        figsize=(15, max_frequency * 2),
        sharex=True,
        sharey=True,
    )

    for freq in range(1, max_frequency + 1):
        for i, shape in enumerate(shapes):
            wave = factory.next_wave(n_points, positive_only=positive_only)
            axs[freq - 1, i].plot(wave.generate())
            if freq == 1:
                axs[freq - 1, i].set_title(f"Shape: {shape}")
            if i == 0:
                axs[freq - 1, i].set_ylabel(f"Frequency: {freq}")
    title = "Positive " if positive_only else "Complete "
    title += "Wave Shapes"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for positive_only in [True, False]:
        fig = plot_shapes(
            ["sin", "square", "sawtooth", "triangle", "gaussian"],
            positive_only=positive_only,
        )
        # save fig in the same directory as the script and show it
        fig_name = "positive_" if positive_only else "complete_"
        fig_name += "wave_shapes.png"
        fig.savefig(Path.parent(__file__) / fig_name)
        plt.show()
        plt.close()
