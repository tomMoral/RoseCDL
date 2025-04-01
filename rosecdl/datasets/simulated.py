import numpy as np
from scipy.signal.windows import tukey
from scipy.stats import norm

from rosecdl.utils.convolution import construct_X
from rosecdl.utils.dictionary import tukey_window
from rosecdl.utils.validation import check_random_state


def simulate_1d(
    n_trials=1,
    n_times=1000,
    n_times_atom=30,
    n_atoms=5,
    d=None,
    random_state=42,
    constant_amplitude=False,
    window=True,
    shapes=["triangle", "square", "sin", "gaussian"],
    p_acti=0.75,
    overlap=False,
    p_contaminate=0,
):
    """Simulate 1D time series data with known atoms and their activations.

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
    d : ndarray of shape (n_atoms, n_times_atom) or None
        Optional pre-defined dictionary of atoms. If None, atoms are generated.
    random_state : int or None
        Seed for reproducible randomization.
    constant_amplitude : bool
        If True, all activations have amplitude 1.
    window : bool
        If True, applies Tukey window to smooth atom edges.
    shapes : list of str
        List of possible shapes for atoms if d is None.
    p_acti : float
        Probability of atom activation per time window.
    overlap : bool
        If True, allows atoms to overlap in time.
    p_contaminate : float
        Proportion of contamination to add.

    Returns
    -------
    X : ndarray of shape (n_trials, n_times)
        Generated time series data.
    ds : ndarray of shape (n_atoms, n_times_atom)
        Dictionary of atoms used to generate data.
    z : ndarray of shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        Activation coefficients.

    """
    rng = check_random_state(random_state)

    # Handle dictionary generation or validation
    if d is None:
        ds = np.zeros((n_atoms, n_times_atom))
        for idx, shape, n_cycles in cycler(n_atoms, n_times_atom, shapes):
            ds[idx, :] = get_atoms(
                shape, n_times_atom, n_cycles=n_cycles, random_state=rng
            )
        ds /= np.linalg.norm(ds, axis=1)[:, None]
        if window:
            ds = ds * tukey_window(n_times_atom)[None, :]
    else:
        # Use provided dictionary
        ds = d.copy()
        n_atoms, n_times_atom = ds.shape

    # Validate number of atoms
    assert (
        n_atoms <= ds.shape[0]
    ), f"ds must be has at least {n_atoms} atoms, got {ds.shape[0]}"
    n_times_atom == ds.shape[1]

    # Randomly select subset of atoms if needed
    if n_atoms < ds.shape[0]:
        kk = np.sort(rng.choice(ds.shape[0], n_atoms, replace=False))
        ds = ds[kk]

    # Generate activations
    z = get_activations2(
        rng,
        (n_atoms, n_trials, n_times - n_times_atom + 1),
        n_times_atom,
        constant_amplitude=constant_amplitude,
        p_acti=p_acti,
        overlap=overlap,
    )

    # Construct data
    X = construct_X(z, ds)

    # Add contamination if requested
    outliers = None
    if p_contaminate > 0:
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
        outliers = X_contaminate != 0
        X += X_contaminate

    # Add noise
    X += 0.01 * rng.randn(*X.shape)

    assert X.shape == (n_trials, n_times)
    assert z.shape == (n_atoms, n_trials, n_times - n_times_atom + 1)
    assert ds.shape == (n_atoms, n_times_atom)

    return X, ds, z, outliers


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


def get_activations2(
    rng,
    shape_z,
    n_times_atom,
    constant_amplitude=False,
    p_acti=0.75,
    overlap=False,
):
    """Generate atom activations with specific density and overlap control.

    Parameters
    ----------
    rng : RandomState
        Random number generator
    shape_z : tuple
        Shape of activation matrix (n_atoms, n_trials, n_times_valid)
    n_times_atom : int
        Length of each atom
    constant_amplitude : bool
        If True, all activations have amplitude 1
    p_acti : float
        Percentage of possible atom support taken
    overlap : bool
        If True, allow atoms overlap

    """
    starts = list()
    n_atoms, n_trials, n_times_valid = shape_z

    n_acti = int(n_times_valid / n_times_atom * p_acti / n_atoms)

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
