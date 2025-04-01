import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey
from scipy.stats import norm
from tqdm import tqdm

from rosecdl.utils.convolution import construct_X_multi
from rosecdl.utils.dictionary import init_dictionary, prox_d, tukey_window
from rosecdl.utils.utils_exp import sort_atoms, sort_list_D
from rosecdl.utils.utils_simulated_waves import WaveFactory
from rosecdl.utils.validation import check_random_state


def split_signal(X, n_splits=1, apply_window=True):
    """Split the signal in ``n_splits`` chunks for faster training.

    This function can be used to accelerate the dictionary learning algorithm
    by creating independent chunks that can be processed in parallel. This can
    bias the estimation and can create border artifacts so the number of chunks
    should be kept as small as possible (`e.g.` equal to ``n_jobs``).

    Also, it is advised to not use the result of this function to
    call the ``DictionaryLearning.transform`` method, as it would return an
    approximate reduction of the original signal in the sparse basis.

    Note that this is a lossy operation, as all chunks will have length
    ``n_times // n_splits`` and the last ``n_times % n_splits`` samples are
    discarded.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times) or (1, n_channels, n_times)
        Signal to be split. It should be a single signal.
    n_splits : int (default: 1)
        Number of splits to create from the original signal. Default is 1.
    apply_window : bool (default: True)
        If set to True (default), a tukey window is applied to each split to
        reduce the border artifacts by reducing the weights of the chunk
        borders.

    Returns
    -------
    X_split: ndarray, shape (n_splits, n_channels, n_times // n_splits)
        The signal splitted in ``n_splits``.
    """
    msg = "This splitting utility is only designed for one multivariate signal"
    if X.ndim == 3:
        assert (
            X.shape[0] == 1
        ), msg + "(1, n_channels, n_times. Found X.shape={}".format(X.shape)
        X = X[0]
    assert X.ndim == 2, msg + " (n_channels, n_times). Found X.ndim={}.".format(X.ndim)

    n_splits = int(n_splits)
    assert n_splits > 0, "The number of splits should be large than 0."

    n_channels, n_times = X.shape
    n_times = n_times // n_splits
    X_split = X[:, : n_splits * n_times].copy()
    X_split = X_split.reshape(n_channels, n_splits, n_times).swapaxes(0, 1)

    # Apply a window to the signal to reduce the border artifacts
    if apply_window:
        X_split *= tukey(n_times, alpha=0.1)[None, None, :]

    return X_split


def generate_z(
    n_trials,
    n_atoms,
    n_times_valid,
    sparsity=0.01,
    positive_only=False,
    method="uniform",
    **kwargs,
):
    """Generate activation vectors for a given (n_trials, n_atoms, support, sparsity).

    An "activation vector" is a 3D array with shape (n_trials, n_atoms, n_times_valid).
    For each trial and atom, it specifies the activation over a series of time points.
    The activation can be generated using one of several methods ('uniform', 'random',
    'constant', 'gaussian'), and the sparsity of the activations can be controlled
    using the 'sparsity' parameter. If 'sparsity' is an integer, it represents the
    number of activations per atom per trial. If it's a float, it represents the
    fraction of nonzero entries in the activation vector.

    Parameters
    ----------
    n_trials : int
        The number of trials.
    n_atoms : int
        The number of atoms.
    n_times_valid : int
        The number of valid time points.
    sparsity : int or float (default=0.01)
        If an integer, the number of activations per atom per trial. If a float, the
        fraction of nonzero entries.
    positive_only : bool, optional (default=False)
        If True, only positive activations are generated.
    method : str, optional (default='uniform')
        The method to generate the activations. Either 'uniform', 'random', 'constant',
        or 'gaussian'.
    **kwargs : dict
        Additional parameters for the methods.
        A random state can be provided with the 'rng' key
        (default=np.random.RandomState()).
        A seed can be provided with the 'seed' key (default=None). In that case, a
        random state will be created, erasing any provided 'rng'.
        For 'uniform', 'low' and 'high' to specify the range (default: [0, 1])
        For 'constant', 'value' to specify the constant value (default=1).
        For 'gaussian', 'mean' and 'std' to specify the parameters of the normal
        distribution (default: 0 and 1).

    Returns
    -------
    np.array
        A 3D array of activation vectors with shape (n_trials, n_atoms, n_times_valid).

    Example
    -------
    Generate activation vectors with 2 trials, 3 atoms, and 100 valid time points,
    using the 'uniform' method and a sparsity of 0.5:

        z = generate_z(2, 3, 100, 0.5, method='uniform', low=0, high=1)

    This will generate a 3D array with shape (2, 3, 100), with half of the entries
    being zero and the others being random values drawn from a uniform distribution
    between 0 and 1.
    """
    # Input validation
    if not isinstance(n_trials, int) or n_trials < 0:
        raise ValueError("n_trials must be a non-negative integer.")

    if not isinstance(n_atoms, int) or n_atoms < 0:
        raise ValueError("n_atoms must be a non-negative integer.")

    if not isinstance(n_times_valid, int) or n_times_valid < 0:
        raise ValueError("n_times_valid must be a non-negative integer.")

    if isinstance(sparsity, int):
        if sparsity < 0 or (n_times_valid > 0 and sparsity > n_times_valid):
            msg = (
                f"If sparsity is an integer, it must be between 0 and n_times_valid, "
                f"got {sparsity}."
            )
            raise ValueError(msg)
    elif isinstance(sparsity, float):
        if sparsity < 0 or sparsity > 1:
            msg = f"If sparsity is a float, it must be between 0 and 1, got {sparsity}."
            raise ValueError(msg)
    else:
        msg = (
            f"sparsity must be either a non-negative integer or a float "
            f"between 0 and 1, got {sparsity}."
        )
        raise ValueError(msg)

    # Set random state from kwargs
    rng = kwargs.get("rng", np.random.RandomState())
    seed = kwargs.get("seed", None)
    if seed is not None:
        rng = np.random.RandomState(seed)

    shape_z = (n_trials, n_atoms, n_times_valid)

    if method == "uniform":
        values = rng.uniform(kwargs.get("low", 0), kwargs.get("high", 1), size=shape_z)
    elif method == "random":
        values = rng.random(size=shape_z)
    elif method == "constant":
        values = np.full(shape_z, kwargs.get("value", 1.0), dtype=np.float64)
    elif method == "gaussian":
        values = rng.normal(kwargs.get("mean", 0), kwargs.get("std", 1), size=shape_z)
    else:
        msg = (
            f"Unknown method '{method}', available methods are 'uniform', f"
            f"'constant', and 'gaussian'."
        )
        raise ValueError(msg)

    # Create a mask array to control the sparsity of the activation vectors.
    if isinstance(sparsity, int) and sparsity >= 1:
        # Generate activations with a fixed number of nonzero entries
        mask = np.zeros(shape_z)
        for i in range(n_trials):
            for j in range(n_atoms):
                active_indices = rng.choice(n_times_valid, size=sparsity, replace=False)
                mask[i, j, active_indices] = 1
    elif isinstance(sparsity, float) and 0 <= sparsity <= 1:
        # Generate activations with a fixed sparsity
        mask = np.random.uniform(size=shape_z) < sparsity
    else:
        msg = (
            f"Sparsity must be either an integer greater or equal to 1, or a float "
            f"between 0 and 1. Got {sparsity}."
        )
        raise ValueError(msg)

    z = values * mask

    if positive_only:
        z = np.abs(z)

    # Ensure that the output array has the correct shape
    assert z.shape == (
        n_trials,
        n_atoms,
        n_times_valid,
    ), f"Output shape {z.shape} does not match expected shape {(n_trials, n_atoms, n_times_valid)}."

    return z


def expand_z(z, n_times_atom):
    """2d to 3d array.

    Transforms a 3D array z to a 2D array, where 1s in z are expanded across
    n_times_atom for each trial, considering all atoms.

    """
    n_trials, n_atoms, n_times_valid = z.shape

    # Determine the required length of the expanded vector
    required_length = n_times_valid + n_times_atom - 1

    # Initialize the expanded vector with zeros
    expanded_z = np.zeros((n_trials, required_length))

    # Set 1s for each atom in each trial
    for trial in range(n_trials):
        for atom in range(n_atoms):
            for time in range(n_times_valid):
                if z[trial, atom, time] > 0:
                    expanded_z[trial, time : time + n_times_atom] = 1

    return expanded_z


def cycler(n_atoms, n_times_atom, shapes):
    """Helper to generate atom parameters"""
    idx = 0
    for n_cycles in range(1, n_times_atom // 2):
        for shape in shapes:
            yield idx, shape, n_cycles
            idx += 1
            if idx >= n_atoms:
                break
        if idx >= n_atoms:
            break


def get_atoms(shape, n_times_atom, zero_mean=True, n_cycles=1, random_state=None):
    """Generate basic atom shapes.

    Parameters
    ----------
    shape : str
        Shape name ('triangle', 'square', 'sin', 'cos', or 'gaussian')
    n_times_atom : int
        Length of atom
    zero_mean : bool
        If True, ensure atom has zero mean
    n_cycles : int
        Number of cycles/repetitions in the atom
    random_state : RandomState
        Random number generator for gaussian atoms
    """
    if shape == "triangle":
        ds = list()
        for idx in range(n_cycles):
            ds.append(np.linspace(0, 1, n_times_atom // (2 * n_cycles)))
            ds.append(ds[-1][::-1])
        d = np.hstack(ds)
        d = np.pad(d, (0, n_times_atom - d.shape[0]), "constant")
    elif shape == "square":
        ds = list()
        for idx in range(n_cycles):
            ds.append(0.5 * np.ones((n_times_atom // (2 * n_cycles))))
            ds.append(-ds[-1])
        d = np.hstack(ds)
        d = np.pad(d, (0, n_times_atom - d.shape[0]), "constant")
    elif shape == "sin":
        d = np.sin(2 * np.pi * n_cycles * np.linspace(0, 1, n_times_atom))
    elif shape == "cos":
        d = np.cos(2 * np.pi * n_cycles * np.linspace(0, 1, n_times_atom))
    elif shape == "gaussian":
        rng = check_random_state(random_state)
        d = np.zeros(n_times_atom)
        xx = np.linspace(0, n_times_atom, n_times_atom)
        means = np.linspace(0, n_times_atom, num=(n_cycles + 1), endpoint=False)[1:]
        weights = rng.choice([-3, -2, -1, 1, 2, 3], size=n_cycles, replace=True)
        for m, w in zip(means, weights):
            d += w * norm.pdf(xx, loc=m, scale=1)

    if zero_mean:
        d -= np.mean(d)

    return d


def generate_atoms(
    n_atoms,
    n_channels,
    n_times_atom,
    n_patterns_per_atom=1,
    method="shapes",
    positive_only=False,
    window=False,
    **kwargs,
):
    """Generate atoms using different waveform shapes or random noise.

    This function generates atoms by cycling through a set of predefined wave shapes,
    or by using a Gaussian distribution. When cycling through wave shapes, the frequency
    of the wave increases each time it is used.

    Parameters
    ----------
    n_atoms : int
        The number of atoms to generate.

    n_channels : int
        The number of channels for each atom.

    n_times_atom : int
        The length of each atom.

    method : str, optional (default='shapes')
        The method to generate atoms. Either 'shapes', 'gaussian', 'random',
            or 'uniform'.
        If 'shapes', atoms will be generated by cycling through a set of predefined
            wave shapes.
        If 'gaussian', atoms will be generated from a Gaussian distribution.
        If 'random', atoms will be generated from a uniform distribution between
            0 and 1.
        If 'uniform', atoms will be generated from a uniform distribution.

    positive_only : bool, optional (default=False)
        If True, only positive atoms will be generated.

    window : bool, optional (default=False)
        If True, atoms will be windowed with a Tukey window.

    **kwargs : dict
        Additional parameters for the generation method.
        A random state can be provided with the 'rng' key.
        A seed can be provided with the 'seed' key. In that case, a random state will
            be created, erasing any provided 'rng'.
        For the 'shapes' method, 'shapes' which is a list of wave types to cycle
            through. Available shapes are 'sin', 'square', 'sawtooth', 'gaussian'
            and 'triangle'.
        For the 'gaussian' method: 'mean' and 'std' (default 0 and 1).
        For the 'uniform' method: 'low' and 'high' (default 0 and 1).

    Returns
    -------
    np.array
        A 3D array of atoms with shape (n_atoms, n_channels, n_times_atom).
    """
    # Validate inputs
    if not isinstance(n_atoms, int) or n_atoms < 1:
        raise ValueError("n_atoms must be a positive integer")
    if not isinstance(n_channels, int) or n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    if not isinstance(n_times_atom, int) or n_times_atom < 1:
        raise ValueError("n_times_atom must be a positive integer")
    if method not in ["shapes", "gaussian", "random", "uniform"]:
        raise ValueError("method must be either 'shapes' or 'gaussian'")
    if not isinstance(positive_only, bool):
        raise ValueError("positive_only must be a boolean")

    # Set random state from kwargs
    rng = kwargs.get("rng", np.random.RandomState())
    seed = kwargs.get("seed", None)
    if seed is not None:
        rng = np.random.RandomState(seed)

    # Ensure that n_patterns_per_atom is not larger than n_channels
    n_patterns_per_atom = min(n_patterns_per_atom, n_channels)
    n_patterns = n_atoms * n_patterns_per_atom

    # Initialize dictionary
    D = np.zeros((n_atoms, n_channels, n_times_atom))

    if method == "shapes":
        shapes = kwargs.get(
            "shapes", ["sin", "square", "sawtooth", "gaussian", "triangle"]
        )
        start_freq = kwargs.get("start_freq", 1)
        wf = WaveFactory(start_freq=start_freq, shapes=shapes)
        # Generate the atoms using the WaveFactory class. This class generates atoms
        # with different waveform shapes by cycling through them.
        # Each time a shape is used, its frequency is increased.
        patterns = np.array(
            [
                wf.next_wave(n_times_atom, positive_only=positive_only).generate()
                for _ in range(n_patterns)
            ]
        )

    elif method == "gaussian":
        mean = kwargs.get("mean", 0)
        std = kwargs.get("std", 1)
        patterns = rng.normal(mean, std, size=(n_patterns, n_times_atom))

    elif method == "random":
        patterns = rng.uniform(0, 1, size=(n_patterns, n_times_atom))

    elif method == "uniform":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 1)
        patterns = rng.uniform(low, high, size=(n_patterns, n_times_atom))

    # Distribute the atom across channels
    for atom_idx in range(n_atoms):
        for j in range(n_patterns_per_atom):
            channel_idx = (atom_idx + j) % n_channels
            pattern_idx = atom_idx * n_patterns_per_atom + j
            D[atom_idx, channel_idx, :] = patterns[pattern_idx]

        # j = i % n_channels
        # D[i, j, :] = atom

    if window:
        # apply Tukey window
        D *= tukey_window(n_times_atom)[None, None, :]

    if positive_only:
        # Ensure all values are positive
        D = np.abs(D)

    # Normalize the dictionary, each atom is then of norm 1
    D = prox_d(D)

    # Ensure that the output array has the correct shape
    assert D.shape == (
        n_atoms,
        n_channels,
        n_times_atom,
    ), f"Output shape {D.shape} does not match expected shape {(n_atoms, n_channels, n_times_atom)}."

    return D


def plot_dicts(*dicts, D_true=None, labels=None, sup_title=None, sort_dicts=True):
    """Plot one or more dictionaries.

    Parameters
    ----------
    *dicts : tuple of np.array
        Dictionaries to be plotted. They should all have the same shape.

    D_true : np.array, default=None
        Ground truth dictionary. If provided, it will be plotted with black
        dashed lines.

    labels : list of str, default=None
        List of labels corresponding to the dictionaries. If not provided,
        labels are not shown on the plot.

    sup_title : str, default=None
        Super title for the entire plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    if not dicts:
        raise ValueError("At least one dictionary should be provided.")

    D = dicts[0]
    n_atoms, n_channels, n_times_atom = D.shape

    for d in dicts:
        if d.shape != D.shape:
            raise ValueError("All dictionaries should have the same shape.")

    if D_true is not None:
        n_atoms_true = D_true.shape[0]
        if n_atoms > n_atoms_true:
            warnings.warn(f"Only plotting the first {n_atoms_true} atoms")
            n_atoms = n_atoms_true

    if sort_dicts:
        dicts = sort_list_D(*dicts, D_ref=D_true)

    if labels is None:
        labels = [None] * len(dicts)

    if len(labels) != len(dicts):
        raise ValueError("Number of labels should match the number of dictionaries.")

    fig, axs = plt.subplots(
        n_atoms, n_channels, figsize=(10, 2 * n_atoms), sharex=True, sharey=True
    )

    # Ensure that axs is always a 2D array, even if n_atoms or n_channels is 1
    if n_atoms == 1 or n_channels == 1:
        axs = np.array(axs).reshape(n_atoms, n_channels)  # reshape to 2D

    for i in range(n_atoms):
        for j in range(n_channels):
            if D_true is not None:
                if i == 0 and j == (n_channels - 1):
                    # Only add legend for top right subplot
                    label = "D_true"
                else:
                    label = None
                axs[i, j].plot(
                    D_true[i, j, :], color="black", linestyle="--", label="D_true"
                )
            for d, label in zip(dicts, labels):
                if i == 0 and j == (n_channels - 1):
                    # Only add legend for top right subplot
                    label = label
                else:
                    label = None
                axs[i, j].plot(d[i, j, :], label=label, alpha=0.7)

            axs[i, j].set_title(f"Atom {i + 1}, Channel {j + 1}")
            if i == n_atoms - 1:
                axs[i, j].set_xlabel("Time")
            if j == n_channels - 1:
                axs[i, j].legend()

    plt.xlim(0, n_times_atom - 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if sup_title:
        fig.suptitle(sup_title)

    return fig


def create_gif_from_dict_lists(
    *list_Ds,
    D_true=None,
    labels=None,
    n_atoms_plot=None,
    sort_dicts=True,
    save_dir=None,
    gif_name="dict_learning.gif",
):
    """Generate a GIF from one or more lists of dictionary representations (D matrices).

    Parameters
    ----------
    *list_Ds : tuple of list of np.array
        Each item is a list of dictionary matrices over iterations.

    D_true : np.array, default=None
        Ground truth D matrix. If provided, it will be used to sort atoms.

    labels : list of str, default=None
        List of labels corresponding to the dictionaries.

    n_atoms_plot : int, default=None
        Number of atoms to plot. If not provided, it will default to the number of
        atoms in the first dictionary in the first list.

    save_dir : str or Path, default=None
        Directory to save the generated GIF. If None, saves to current
        directory.

    gif_name : str, default="dict_learning.gif"
        Name of the GIF file to be saved.

    Returns
    -------
    gif_path : Path
        Path to the created GIF.
    """
    import imageio

    # Convert list_Ds into numpy array
    array_Ds = np.array(list_Ds)
    n_dicts, n_iter, n_atoms, _, _ = array_Ds.shape

    if n_atoms_plot is None:
        n_atoms_plot = n_atoms
    n_atoms_plot = min(n_atoms_plot, n_atoms)

    # Convert save_dir to Path instance if it's a string
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    elif save_dir is None:
        save_dir = Path.cwd()

    save_dir.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = [f"Dict {i}" for i in range(n_dicts)]

    if D_true is not None:
        n_atoms_plot = min(n_atoms_plot, D_true.shape[0])
        D_true = D_true[:n_atoms_plot]

    if sort_dicts:
        array_Ds_sorted = []
        if D_true is not None:
            D_ref = D_true
        else:
            # If no reference is given, take the final dictionary of the first list
            D_ref = array_Ds[0][-1]

        for this_list_D in array_Ds:
            _, permutation = sort_atoms(this_list_D[-1], D_ref, return_permutation=True)
            this_list_D_sorted = [D[permutation] for D in this_list_D]
            array_Ds_sorted.append(this_list_D_sorted)

        array_Ds = np.array(array_Ds_sorted)

    # List to store the paths of the image files
    image_files = []

    for i in tqdm(range(n_iter)):
        current_dicts = [array_Ds[j, i] for j in range(n_dicts)]

        # Generate plot
        plot_dicts(
            *current_dicts,
            D_true=D_true,
            labels=labels,
            sup_title=f"Iteration {i + 1}/{n_iter}",
            sort_dicts=False,
        )

        # Save plot as image file
        this_image_file = save_dir / f"plot_{i}.png"
        plt.savefig(this_image_file)
        image_files.append(this_image_file)

        # Close the plot to free memory
        plt.close()

    # Create gif
    images = [imageio.imread(file) for file in image_files]
    imageio.mimsave(save_dir / gif_name, images, fps=5)  # fps controls the speed

    # Optionally, delete the image files after creating the gif
    for file in image_files:
        file.unlink()

    return save_dir / gif_name


def generate_signal(
    n_trials,
    n_channels,
    n_times,
    n_atoms,
    n_times_atom,
    n_patterns_per_atom=1,
    init_d="shapes",
    init_d_kwargs={},
    sparsity=0.05,
    init_z="uniform",
    init_z_kwargs={"low": 0, "high": 1},
    window=False,
    rng=None,
):
    """Generate a multi-channel signal based on a set of atoms and activations.

    This function generates a 3D array of signals, where the first dimension
    corresponds to different trials, the second dimension corresponds to
    different channels, and the third dimension corresponds to different time
    points. The signal is generated by convolving a set of atoms (generated by
    the `generate_atoms` function) with a set of activations (generated by the
    `generate_z` function).

    Parameters
    ----------
    n_trials : int
        The number of trials. Must be a positive integer.
    n_channels : int
        The number of channels. Must be a positive integer.
    n_times : int
        The length of the signal. Must be a positive integer.
    n_atoms : int
        The number of atoms to generate. Must be a positive integer.
    n_times_atom : int
        The length of each atom. Must be a positive integer.
    n_patterns_per_atom : int
        The number of patterns per atom. Must be a positive integer.
    init_d : str
        The method to generate atoms. See `generate_atoms` for details.
    init_d_kwargs : dict
        Additional keyword arguments for `generate_atoms`.
    sparsity : float
        The sparsity of the activations. See `generate_z` for details.
    init_z : str
        The method to generate activations. See `generate_z` for details.
    init_z_kwargs : dict
        Additional keyword arguments for `generate_z`.
    window : bool
        Whether to window the atoms. See `generate_atoms` for details.
    rng : int, RandomState instance or None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        The generated signal.

    Examples
    --------
    >>> from benchmark_utils.generate_signal import generate_signal
    >>> X = generate_signal(
    ...     n_trials=2,
    ...     n_channels=3,
    ...     n_times=100,
    ...     n_atoms=2,
    ...     n_times_atom=10,
    ...     init_d="uniform",
    ...     init_d_kwargs={"low": 0, "high": 1},
    ...     sparsity=0.5,
    ...     init_z="uniform",
    ...     init_z_kwargs={"low": 0, "high": 1},
    ...     window=False,
    ...     rng=0,
    ... )
    >>> X.shape
    (2, 3, 100)
    """
    rng = check_random_state(rng)

    # Generate atoms
    D = generate_atoms(
        n_atoms=n_atoms,
        n_channels=n_channels,
        n_times_atom=n_times_atom,
        n_patterns_per_atom=n_patterns_per_atom,
        method=init_d,
        positive_only=False,
        window=window,
        rng=rng,
        **init_d_kwargs,
    )

    # Generate activations
    z = generate_z(
        n_trials=n_trials,
        n_atoms=n_atoms,
        n_times_valid=n_times - n_times_atom + 1,
        sparsity=sparsity,
        positive_only=False,
        method=init_z,
        rng=rng,
        **init_z_kwargs,
    )

    # Construct X
    X = construct_X_multi(z, D=D, n_channels=n_channels)

    assert X.shape == (
        n_trials,
        n_channels,
        n_times,
    ), f"Output shape {X.shape} does not match expected shape {(n_trials, n_channels, n_times)}."

    return X, z, D


def validate_sparsity(sparsity, n_trials):
    if isinstance(sparsity, float) and 0 <= sparsity <= 1:
        return int(n_trials * sparsity)
    elif isinstance(sparsity, int) and sparsity >= 1:
        return sparsity
    else:
        raise ValueError(
            f"Sparsity must be either an integer greater or equal to 1, or a float "
            f"between 0 and 1. Got {sparsity}."
        )


def apply_contamination(
    X, contamination_params=None, rng=None, n_times_atom=64, n_splits=1, verbose=0
):
    """Add optional contamination to the signal.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
        The original signal to be contaminated.

    contamination_params : dict or None, optional (default=None)
        Parameters to define the contamination. If None, no contamination is applied.

    rng : int, RandomState instance or None, optional (default=None)
        Random number generator or seed for reproducibility.

    n_times_atom : int, optional (default=64)
        Number of times for the atom. Used when adding contaminated atoms.

    Returns
    -------
    X : np.ndarray
        The contaminated signal.
    """
    if contamination_params is None:
        info = dict(percentage=0)
        return X, info

    rng = check_random_state(rng)
    n_trials, n_channels, n_times = X.shape
    contaminate_trial = contamination_params.get("contaminate_trials", False)

    if contaminate_trial:
        # Contamination is applied to the whole trial(s)
        sparsity = validate_sparsity(contamination_params.get("sparsity", 1), n_trials)

        contam_indices = rng.choice(n_trials, size=sparsity, replace=False)
        print(f"Contaminate {sparsity} trials")

        X[contam_indices, :, :] = rng.uniform(
            X.min(), X.max(), size=(sparsity, n_channels, n_times)
        )
    else:  # add contamined atoms
        init_d = contamination_params.get("init_d", "random")
        init_d_kwargs = contamination_params.get("init_d_kwargs", {})
        if init_d == "random":
            init_d = "uniform"
            init_d_kwargs.setdefault("low", X.min())
            init_d_kwargs.setdefault("high", X.max())

        if init_d == "uniform":
            if verbose:
                print(
                    "Add uniform contamination between {:.3f} and {:.3f}.".format(
                        init_d_kwargs["low"], init_d_kwargs["high"]
                    )
                )

        sparsity = contamination_params.get("sparsity", 1)
        if sparsity >= 1:
            # if sparsity is an integer, it is the number of non-zero atoms
            sparsity *= n_splits

        # Generate contamination
        X_contam, z_contam, D_contam = generate_signal(
            n_trials,
            n_channels,
            n_times,
            n_atoms=contamination_params.get("n_atoms", 1),
            n_times_atom=contamination_params.get("n_times_atom", 3 * n_times_atom),
            n_patterns_per_atom=contamination_params.get("n_patterns_per_atom", 1),
            init_d=init_d,
            init_d_kwargs=init_d_kwargs,
            sparsity=sparsity,
            init_z=contamination_params.get("init_z", "constant"),
            init_z_kwargs=contamination_params.get("init_z_kwargs", {"value": 1}),
            window=True,
            rng=rng,
        )
        # Reshuflle the contamination so each contamination atom is different
        mask = X_contam != 0
        unif_contam = np.random.uniform(
            X_contam.min(), X_contam.max(), size=X_contam.shape
        )
        X_contam = unif_contam * mask
        # Add contamination to the signal
        X += X_contam

    outliers_mask = X_contam != 0

    # Compute percentage of contamination
    percentage = 100 * np.count_nonzero(X_contam) / X.size
    if verbose:
        print(f"Add {percentage:.2f}% contamination.")

    info = dict(
        percentage=percentage,
        X_contam=X_contam,
        z_contam=z_contam,
        D_contam=D_contam,
        outliers_mask=outliers_mask,
    )

    return X, info


def generate_experiment(
    simulation_params,
    D_init_shape=None,
    save_dir=None,
    verbose=0,
    return_info_contam=False,
):
    """Generate a synthetic signal experiment with optional contamination.

    Parameters
    ----------
    simulation_params : dict
        Dictionary containing the parameters for the simulation, including:
        - n_trials : int, Number of trials. Optional, default is 1.
        - n_channels : int, Number of channels. Optional, default is 1.
        - n_times : int, Number of time points.
        - n_atoms : int, Number of atoms.
        - n_atom_extra : int, Extra atoms for noise learning.
        - n_times_atom : int, Duration of each atom.
        - n_patterns_per_atom : int, Number of patterns per atom.
        - n_iter, solver_z_max_iter, solver_z_tol, reg : Various solver parameters.
        - rng : Random number generator, instance of RandomState.
            Optional, default is None.
        - init_d
        - init_d_kwargs
        - sparsity
        - init_z
        - init_z_kwargs
        - window
        - contamination_params : Optional parameters for contamination.

    save_dir : str, optional
        Directory to save the generated X, z, and D. If None, the data is not saved.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Generated signal.
    z : ndarray, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        Sparse activations.
    D : ndarray, shape (n_atoms, n_channels, n_times_atom)
        Generated dictionary.
    D_init : ndarray, shape (n_atoms + n_atom_extra, n_channels, n_times_atom)
        Initialized dictionary.
    """
    # Check required parameters
    if (
        "n_times" not in simulation_params
        or "n_atoms" not in simulation_params
        or "n_times_atom" not in simulation_params
    ):
        raise ValueError("n_times, n_atoms, and n_times_atom must be specified.")

    # Extract parameters
    n_trials = simulation_params.get("n_trials", 1)
    n_channels = simulation_params.get("n_channels", 1)
    n_times = simulation_params["n_times"]
    n_atoms = simulation_params["n_atoms"]
    n_times_atom = simulation_params["n_times_atom"]
    window = simulation_params.get("window", True)
    rng = check_random_state(simulation_params.get("rng", None))
    sparsity = simulation_params.get("sparsity", 1)
    rank1 = simulation_params.pop("rank1", False)

    #
    n_times *= n_trials
    n_splits = n_trials
    if sparsity >= 1:
        # if sparsity is an integer, it is the number of non-zero atoms
        sparsity *= n_trials

    n_trials = 1

    # Generate signal
    if verbose:
        print("Generating signal... ", end="", flush=True)
    X, z, D = generate_signal(
        n_trials,
        n_channels,
        n_times,
        n_atoms,
        n_times_atom,
        n_patterns_per_atom=simulation_params.get("n_patterns_per_atom", 1),
        init_d=simulation_params.get("init_d", "shapes"),
        init_d_kwargs=simulation_params.get(
            "init_d_kwargs", {"shapes": ["sin", "triangle"]}
        ),
        sparsity=sparsity,
        init_z=simulation_params.get("init_z", "constant"),
        init_z_kwargs=simulation_params.get("init_z_kwargs", {"value": 1}),
        window=window,
        rng=rng,
    )
    # Add noise
    X += rng.normal(scale=simulation_params.get("noise_std", 0.01), size=X.shape)

    # Normalize
    X /= X.std()

    # Optional contamination
    X_clean = X.copy()
    contamination_params = simulation_params.get("contamination_params", None)
    X, info_contam = apply_contamination(
        X,
        contamination_params=contamination_params,
        rng=rng,
        n_times_atom=n_times_atom,
        n_splits=n_splits,
        verbose=verbose,
    )

    if verbose:
        print("done")

    # Split signal
    X = split_signal(X, n_splits=n_splits, apply_window=True)
    z = split_signal(z, n_splits=n_splits, apply_window=False)

    if contamination_params is not None:
        info_contam["z_contam"] = split_signal(
            info_contam["z_contam"], n_splits=n_splits, apply_window=False
        )
        info_contam["outliers_mask"] = split_signal(
            info_contam["outliers_mask"], n_splits=n_splits, apply_window=False
        )
        info_contam["X_clean"] = split_signal(
            X_clean, n_splits=n_splits, apply_window=True
        )
        info_contam["X_contam"] = split_signal(
            info_contam["X_contam"], n_splits=n_splits, apply_window=True
        )

    # Initialize the dictionary with chunks of the signal
    if D_init_shape is None:
        D_init_shape = (
            n_atoms + simulation_params.get("n_atoms_extra", 0),
            n_channels,
            n_times_atom,
        )

    D_init = init_dictionary(
        X,
        D_init_shape[0],
        D_init_shape[2],
        rank1=rank1,
        window=window,
        D_init=simulation_params.get("D_init", "chunk"),
        random_state=rng,
    )

    # Save data if save_dir is specified
    if save_dir:
        exp_dir = Path(save_dir)
        np.save(exp_dir / "X.npy", X)
        np.save(exp_dir / "z.npy", z)
        np.save(exp_dir / "D.npy", D)
        np.save(exp_dir / "D_init.npy", D_init)

    if return_info_contam:
        return X, z, D, D_init, info_contam

    return X, z, D, D_init


def plot_signal(*list_X, X_true=None, labels=None, label_true="Original"):
    """Plot a multi-channel signal.

    Parameters
    ----------
    list_X : ndarray, shape (n_signals, n_trials, n_channels, n_times)
        The signals to plot.

    X_true : ndarray, shape (n_trials, n_channels, n_times), optional, default: None
        The true signal.

    Returns
    -------
    fig : matplotlib figure
    """
    if not list_X:
        raise ValueError("At least one signal should be provided.")

    list_X = np.array(list_X)
    n_signals, n_trials, n_channels, n_times = list_X.shape
    X = list_X[0]

    for x in list_X:
        if x.shape != X.shape:
            raise ValueError("All signals should have the same shape.")

    if (X_true is not None) and (not X_true.shape == X.shape):
        raise ValueError(
            "X_true should have the same shape as the other signals. "
            f"X_true is of shape {X_true.shape}, other signals are of shape {X.shape}."
        )

    if labels is None:
        labels = [None] * n_signals

    if len(labels) != n_signals:
        raise ValueError("Number of labels should match the number of signals.")

    if n_trials == 1 and n_channels == 1:
        fig, axs = plt.subplots(figsize=(10, 2 * n_trials))
        axs = np.array([[axs]])  # make axs 2D
    elif n_trials == 1 or n_channels == 1:
        fig, axs = plt.subplots(
            max(n_trials, n_channels),
            figsize=(10, 2 * n_trials),
            sharex=True,
            sharey=True,
        )
        axs = axs.reshape(n_trials, n_channels)  # reshape to 2D
    else:
        fig, axs = plt.subplots(
            n_trials, n_channels, figsize=(10, 2 * n_trials), sharex=True, sharey=True
        )

    if n_signals == 1 and X_true is None:
        alpha = 1
    else:
        alpha = 0.6

    for i in tqdm(range(n_trials)):
        for j in range(n_channels):
            if X_true is not None:
                axs[i, j].plot(
                    X_true[i, j, :],
                    color="black",
                    linestyle="--",
                    alpha=alpha,
                    label=label_true,
                )

            for X, label in zip(list_X, labels):
                axs[i, j].plot(X[i, j, :], alpha=alpha, label=label)

            axs[i, j].set_title(f"Trial {i + 1}, Channel {j + 1}")
            if j == 0:
                axs[i, j].set_ylabel("Amplitude")
            if j == n_channels - 1 and label is not None:
                axs[i, j].legend()
            if i == n_trials - 1:
                axs[i, j].set_xlabel("Time")

    plt.xlim(0, n_times)
    fig.tight_layout()
    return fig
