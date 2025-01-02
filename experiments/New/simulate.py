# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
import matplotlib.pyplot as plt
import numpy as np
from alphacsc.utils.convolution import construct_X, construct_X_multi
from alphacsc.utils.dictionary import tukey_window
from alphacsc.utils.validation import check_random_state
from scipy.signal.windows import tukey
from scipy.stats import norm


def simulate_data(
    n_trials=1,
    n_channels=1,
    n_times=1_000,
    n_times_atom=30,
    n_atoms=5,
    n_acti_atom=1,
    random_state=42,
    constant_amplitude=False,
    window=True,
    shapes=["triangle", "square", "sin", "gaussian"],
    sigma_noise=0,
    plot_atoms=False,
):
    """Simulate the data.

    Parameters
    ----------
    n_trials : int
        Number of samples / trials.
    n_channels : int
        Number of channels in the signals
    n_times : int
        Number of time points.
    n_times_atom : int
        Number of time points.
    n_atoms : int
        Number of atoms.
    random_state : int | None
        If integer, fix the random state.
    constant_amplitude : float
        If True, the activations have constant amplitude.

    Returns
    -------
    # if n_channels = 1:
    #     X : array, shape (n_trials, n_times)
    #         The data
    #     ds : array, shape (n_atoms, n_times_atom)
    #         The true atoms.
    #     z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
    #         The true codes.
    # else:
    X : array, shape (n_trials, n_channels, n_times)
        The data
    D : array, shape (n_atoms, n_channels, n_times_atom)
        The true atoms.
    z : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        The true codes.

    Note
    ----
    X will be non-zero from n_times_atom to n_times.
    """
    # add atoms
    rng = check_random_state(random_state)

    ds = np.zeros((n_atoms, n_times_atom))
    for idx, shape, n_cycles in cycler(n_atoms, n_times_atom, shapes):
        ds[idx, :] = get_atoms(shape, n_times_atom, n_cycles=n_cycles, random_state=rng)
    ds /= np.linalg.norm(ds, axis=1)[:, None]
    if window:
        ds = ds * tukey_window(n_times_atom)[None, :]
    if plot_atoms:
        fig, axes = plt.subplots(
            1, n_atoms, figsize=(3 * n_atoms, 3), sharex=True, sharey=True
        )
        for ax, this_d in zip(axes, ds):
            ax.plot(this_d)
            ax.set_xlim(0, n_times_atom - 1)
        plt.show()

    D = np.zeros((n_atoms, n_channels, n_times_atom))
    for k in range(n_atoms):
        i_channel = k % n_channels
        D[k, i_channel] = ds[k, :]

    z = get_activations_multi(
        rng,
        (n_trials, n_atoms, n_times - n_times_atom + 1),
        n_acti_atom=n_acti_atom,
        constant_amplitude=constant_amplitude,
    )
    X = construct_X_multi(z, D=D, n_channels=n_channels)
    if sigma_noise > 0:
        X += rng.normal(scale=sigma_noise, size=X.shape)

    # if n_channels == 1:
    #     assert X.shape == (n_trials, n_times)
    #     assert z.shape == (n_atoms, n_trials, n_times - n_times_atom + 1)
    #     assert ds.shape == (n_atoms, n_times_atom)
    #     return X, ds, z
    # else:
    assert X.shape == (n_trials, n_channels, n_times)
    assert z.shape == (n_trials, n_atoms, n_times - n_times_atom + 1)
    assert D.shape == (n_atoms, n_channels, n_times_atom)
    return X, D, z


def cycler(n_atoms, n_times_atom, shapes=["triangle", "square", "sin"]):
    idx = 0
    for n_cycles in range(1, n_times_atom // 2):
        for shape in shapes:
            yield idx, shape, n_cycles
            idx += 1
            if idx >= n_atoms:
                break
        if idx >= n_atoms:
            break


def get_activations(rng, shape_z, n_acti_atom=1, constant_amplitude=False):
    starts = list()
    n_atoms, n_trials, n_times_valid = shape_z
    for idx in range(n_atoms):
        starts.append(
            rng.randint(low=0, high=n_times_valid, size=(n_trials, n_acti_atom))
        )
    # add activations
    z = np.zeros(shape_z)
    for i in range(n_trials):
        for k_idx, start in enumerate(starts):
            if constant_amplitude:
                randnum = np.ones_like(starts[k_idx][i])
            else:
                randnum = rng.uniform(size=starts[k_idx][i].shape)
            z[k_idx, i, starts[k_idx][i]] = randnum
    return z


def get_activations_multi(rng, shape_z, n_acti_atom=1, constant_amplitude=False):
    """
    n_acti_atom : int
        Number of activation per atom.
    """
    starts = list()
    n_trials, n_atoms, n_times_valid = shape_z
    for idx in range(n_trials):
        starts.append(
            rng.randint(low=0, high=n_times_valid, size=(n_atoms, n_acti_atom))
        )
    # add activations
    z = np.zeros(shape_z)
    for i_trial, trial_starts in enumerate(starts):
        for atom_id, atom_starts in enumerate(trial_starts):
            if constant_amplitude:
                randnum = np.ones_like(atom_starts)
            else:
                randnum = rng.uniform(size=atom_starts.shape)
            z[i_trial, atom_id, atom_starts] = randnum
    return z


def get_atoms(shape, n_times_atom, zero_mean=True, n_cycles=1, random_state=None):
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
        # evenly spaced gaussian means
        means = np.linspace(0, n_times_atom, num=(n_cycles + 1), endpoint=False)[1:]
        # random weights uniformy drawned in [-3, 3]
        weights = rng.choice([-3, -2, -1, 1, 2, 3], size=n_cycles, replace=True)
        for m, w in zip(means, weights):
            d += w * norm.pdf(xx, loc=m, scale=1)

    if zero_mean:
        d -= np.mean(d)

    return d


def get_activations2(
    rng,
    shape_z,
    n_times_atom,
    constant_amplitude=False,
    p_acti=0.75,
    overlap=False,
):
    """
    p_acti : float between 0 and 1
        Percentage of possible atom support taken

    overlap : bool
        If True, allow atoms overlap
    """
    starts = list()
    n_atoms, n_trials, n_times_valid = shape_z

    n_acti = int(n_times_valid / n_times_atom * p_acti / n_atoms)
    print(f"{n_acti} activations per atom per trial")

    # add activations
    z = np.zeros(shape_z)
    for i in range(n_trials):
        starts = rng.choice(
            range(n_times_valid), size=(n_atoms, n_acti), replace=overlap
        )
        for k_idx, start in enumerate(starts):
            if constant_amplitude:
                randnum = 1.0
            else:
                randnum = rng.uniform()
            z[k_idx, i, starts[k_idx]] = randnum

    return z


def simulate_data2(
    n_trials=1,
    n_times=1_000,
    n_times_atom=30,
    n_atoms=5,
    random_state=42,
    constant_amplitude=False,
    window=True,
    shapes=["triangle", "square", "sin", "gaussian"],
    plot_atoms=False,
    p_acti=0.75,
    overlap=False,
    p_contaminate=0,
):
    """Simulate data with known atoms and their activations.

    This function generates synthetic time series data by creating a set of basic
    shapes (atoms) and their temporal activations. It can optionally add noise
    and contamination.

    Parameters
    ----------
    n_trials : int
        Number of samples to generate.
    n_times : int
        Length of each sample.
    n_times_atom : int
        Length of each atom (temporal pattern).
    n_atoms : int
        Number of distinct atoms to generate.
    random_state : int or None, default=42
        Seed for reproducible randomization. If None, randomization is not controlled.
    constant_amplitude : bool, default=False
        If True, all activations have amplitude 1. If False, amplitudes are random.
    window : bool, default=True
        If True, applies a Tukey window to smooth atom edges.
    shapes : list of str, default=["triangle", "square", "sin", "gaussian"]
        List of possible shapes for the atoms. Each atom will use one of these shapes.
    plot_atoms : bool, default=False
        If True, displays a plot of the generated atoms.
    p_acti : float, default=0.75
        Probability of atom activation per time window.
    overlap : bool, default=False
        If True, allows atoms to overlap in time.
    p_contaminate : float, default=0
        Proportion of contamination to add. Must be between 0 and 1.

    Returns
    -------
    X : ndarray of shape (n_trials, n_times)
        The generated time series data.
    ds : ndarray of shape (n_atoms, n_times_atom)
        The dictionary of atoms used to generate the data.
    z : ndarray of shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The activation coefficients for each atom.

    Notes
    -----
    The function generates data following the model:
    X = sum(d_k * z_k) + noise + contamination
    where d_k are the atoms and z_k their activations.

    Examples
    --------
    >>> X, ds, z = simulate_data2(n_trials=10, n_times=1000, n_times_atom=30, n_atoms=3)
    >>> print(f"Data shape: {X.shape}")
    Data shape: (10, 1, 1000)
    """
    # Initialize random number generator
    rng = check_random_state(random_state)

    # Generate the basic atoms/shapes
    ds = np.zeros((n_atoms, n_times_atom))
    for idx, shape, n_cycles in cycler(n_atoms, n_times_atom, shapes):
        ds[idx, :] = get_atoms(shape, n_times_atom, n_cycles=n_cycles, random_state=rng)

    # Normalize atoms and apply window if requested
    ds /= np.linalg.norm(ds, axis=1)[:, None]
    if window:
        ds = ds * tukey_window(n_times_atom)[None, :]

    # Plot atoms if requested
    if plot_atoms:
        fig, axes = plt.subplots(
            1, n_atoms, figsize=(3 * n_atoms, 3), sharex=True, sharey=True
        )
        for ax, this_d in zip(axes, ds):
            ax.plot(this_d)
            ax.set_xlim(0, n_times_atom - 1)
        plt.show()

    # Validate number of atoms
    assert (
        n_atoms <= ds.shape[0]
    ), f"ds must be has at least {n_atoms} atoms, got {ds.shape[0]}"
    n_times_atom == ds.shape[1]

    # Randomly select subset of atoms if needed
    if n_atoms < ds.shape[0]:
        kk = np.sort(rng.choice(ds.shape[0], n_atoms, replace=False))
        ds = ds[kk]

    # Generate activation coefficients
    z = get_activations2(
        rng,
        (n_atoms, n_trials, n_times - n_times_atom + 1),
        n_times_atom,
        constant_amplitude=constant_amplitude,
        p_acti=p_acti,
        overlap=overlap,
    )

    # Construct the data by convolving atoms with activations
    X = construct_X(z, ds)

    # Add contamination if requested
    if p_contaminate > 0:
        # create contamination atom
        n_times_atom_contaminate = 3 * n_times_atom
        atom_noise = np.random.uniform(
            ds.min(), ds.max(), (1, n_times_atom_contaminate)
        )
        atom_noise *= tukey(n_times_atom_contaminate, alpha=0.2)

        z_contaminate = get_activations2(
            rng,
            (1, n_trials, n_times - n_times_atom_contaminate + 1),
            n_times_atom_contaminate,
            constant_amplitude=constant_amplitude,
            p_acti=p_contaminate,
            overlap=overlap,
        )

        X_contaminate = construct_X(z_contaminate, atom_noise)
        X += X_contaminate

    # Add small Gaussian noise
    X += 0.01 * rng.randn(*X.shape)

    # Validate output shapes
    assert X.shape == (n_trials, n_times)
    assert z.shape == (n_atoms, n_trials, n_times - n_times_atom + 1)
    assert ds.shape == (n_atoms, n_times_atom)

    return X, ds, z
