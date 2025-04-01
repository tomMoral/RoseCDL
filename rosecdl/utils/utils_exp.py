import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


def multi_channel_pearson_coef(atom, pat):
    """Compute the multi-channel pearson coefficient per patch between an atom and
    a reference pattern.

    Parameters
    ----------
        atom, pat : np.ndarray
            ND array of shape (n_channels, *atom_support).

    Returns
    -------
        np.ndarray : The pearson coefficient per patch.

    """
    patch = np.ones(np.minimum(atom.shape[1:], pat.shape[1:]))
    N = atom.shape[0] * np.prod(patch.shape)
    corr = (
        np.sum(
            [
                signal.correlate(atom_c, pat_c, mode="same")
                for atom_c, pat_c in zip(atom, pat, strict=False)
            ],
            axis=0,
        )
        / N
    )
    mean_atom = (
        np.sum(
            [signal.correlate(atom_c, patch, mode="valid") for atom_c in atom], axis=0
        )
        / N
    )
    norm_atom = (
        np.sum(
            [signal.correlate(atom_c**2, patch, mode="valid") for atom_c in atom],
            axis=0,
        )
        / N
    )
    mean_pat = (
        np.sum([signal.correlate(pat_c, patch, mode="valid") for pat_c in pat], axis=0)
        / N
    )
    norm_pat = (
        np.sum(
            [signal.correlate(pat_c**2, patch, mode="valid") for pat_c in pat], axis=0
        )
        / N
    )
    mean = mean_atom * mean_pat
    norm = np.maximum(
        np.sqrt((norm_atom - mean_atom**2) * (norm_pat - mean_pat**2)), 1e-6
    )

    padding = tuple(
        ((p - 1) // 2 + ((p - 1) % 2 == 1), (p - 1) // 2) for p in patch.shape
    )
    mean = np.pad(mean, padding, "edge")
    norm = np.pad(norm, padding, "edge")

    return (corr - mean) / norm


def evaluate_D_hat(patterns, D_hat):
    """Evaluate the learned dictionary D_hat with respect to a set of patterns.

    Parameters
    ----------
        patterns : np.ndarray
            The set of patterns, either:
            - 4D array (n_patterns, n_channels, height, width) for 2D images
            - 3D array (n_patterns, n_channels, n_times_atom) for 1D signals
        D_hat : np.ndarray
            The learned dictionary, same shape as patterns

    Returns
    -------
        float : The evaluation score (mean correlation of best assignments).

    """
    corr = np.array(
        [
            [multi_channel_pearson_coef(dk, pat).max() for dk in D_hat]
            for pat in patterns
        ]
    )

    i, j = linear_sum_assignment(corr, maximize=True)
    return corr[i, j].mean()


def check_and_load_data(exp_dir, file_name):
    file_path = Path(exp_dir) / file_name
    if file_path.exists():
        return np.load(file_path)
    return None


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def check_and_initialize(
    exp_dir, simulation_params, rosecdl_params, results_file_name="df_results"
):
    """Checks if previous simulation parameters exist and initializes the environment accordingly.

    Parameters
    ----------
    - exp_dir (Path): Directory where experiment results are stored.
    - simulation_params (dict): Current simulation parameters.
    - rosecdl_params (dict): Current rosecdl parameters.
    - results_file_name (str): Name of the CSV file to store results.

    Returns
    -------
    - df_results (DataFrame): DataFrame to hold results, either loaded from file or initialized as empty.

    """
    # Define paths based on exp_dir
    simulation_path = exp_dir / "simulation_params.json"
    rosecdl_path = exp_dir / "rosecdl_params.json"
    results_path = exp_dir / f"{results_file_name}.csv"

    # Check if parameter files exist and compare them with current parameters
    if simulation_path.exists() and rosecdl_path.exists():
        previous_simulation_params = load_json(simulation_path)
        previous_rosecdl_params = load_json(rosecdl_path)
        if (
            simulation_params != previous_simulation_params
            or rosecdl_params != previous_rosecdl_params
        ):
            print("Parameters changed, starting from scratch")
            df_results = pd.DataFrame()
            df_results.to_csv(results_path, index=False)
            save_json(simulation_params, simulation_path)
            save_json(rosecdl_params, rosecdl_path)
        else:
            print("Parameters are the same, loading previous results")
            try:
                df_results = pd.read_csv(results_path)
            except pd.errors.EmptyDataError:
                print("No previous results found, starting from scratch")
                df_results = pd.DataFrame()

            n_runs_done = len(df_results)
            print(f"{n_runs_done} runs already done")
    else:
        print("No previous results found, starting from scratch")
        df_results = pd.DataFrame()
        df_results.to_csv(results_path, index=False)
        save_json(simulation_params, simulation_path)
        save_json(rosecdl_params, rosecdl_path)

    return df_results


def get_method_name(method_spec: dict[str or float]) -> str:
    """Convert method specification to string format."""
    if method_spec["name"] == "none":
        return "no detection"

    return f"{method_spec['name']} (alpha={method_spec['alpha']:.02f})"


def get_outliers_metric(
    true_outliers_mask, rosecdl, X, dice_score_epsilon: float = 1e-7
):
    """rosecdl: RoseCDL instance."""
    # Converting true_outliers_mask to numpy array if not already
    if isinstance(true_outliers_mask, torch.Tensor):
        true_outliers_mask = true_outliers_mask.detach().cpu().numpy()
        true_outliers_mask = true_outliers_mask.astype(np.int32)

    assert isinstance(true_outliers_mask, np.ndarray)

    X = torch.tensor(X, dtype=rosecdl.dtype, device=rosecdl.device)
    X_hat, z_hat = rosecdl.csc(X)

    outliers_mask = rosecdl.loss_fn.get_outliers_mask(X_hat, z_hat, X, opening=False)
    outliers_mask = outliers_mask.detach().cpu().numpy()

    # Ensure masks have the same shape
    if true_outliers_mask.shape != outliers_mask.shape:
        raise ValueError(
            f"Shape of true_outliers_mask ({true_outliers_mask.shape}) "
            f"does not match shape of outliers_mask ({outliers_mask.shape})"
        )

    # Ensure masks have the same dtype
    if true_outliers_mask.dtype != outliers_mask.dtype and outliers_mask.dtype == bool:
        outliers_mask = outliers_mask.astype(np.int32)

    accuracy = np.mean(outliers_mask == true_outliers_mask)
    precision = precision_score(true_outliers_mask.flatten(), outliers_mask.flatten())
    recall = recall_score(true_outliers_mask.flatten(), outliers_mask.flatten())
    f1 = f1_score(true_outliers_mask.flatten(), outliers_mask.flatten())
    # Compute dice score
    dice = 2 * (precision * recall) / (precision + recall + dice_score_epsilon)
    # Compute Jacard score using sklearn
    jaccard = jaccard_score(true_outliers_mask.flatten(), outliers_mask.flatten())

    score_dict = dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        dice=dice,
        jaccard=jaccard,
        percentage=np.mean(outliers_mask),
    )

    return score_dict


def plot_dicts(*dicts, D_true=None, labels=None, sup_title=None, sort_dicts=True):
    """Plot one or more dictionaries, with the option of overlaying a ground truth dictionary.

    Parameters
    ----------
    *dicts : tuple of np.array
        Dictionaries to be plotted. They should all have the same shape.

    D_true : np.array, default=None
        Ground truth dictionary. If provided, it will be plotted with black dashed lines.

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
                    D_true[i, j, :],
                    color="black",
                    linestyle="--",
                    label="D_true",
                )
            for d, label in zip(dicts, labels, strict=False):
                if i == 0 and j == (n_channels - 1):
                    # Only add legend for top right subplot
                    label = label
                else:
                    label = None
                axs[i, j].plot(d[i, j, :], label=label, alpha=0.7)

            axs[i, j].set_title(f"Atom {i + 1}, Channel {j + 1}")
            if i == n_atoms - 1:
                axs[i, j].set_xlabel("Time")
            if j == n_channels - 1 and labels[0] is not None:
                axs[i, j].legend()

    plt.xlim(0, n_times_atom - 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if sup_title:
        fig.suptitle(sup_title)

    return fig


def sort_list_D(*list_D, D_ref=None):
    """Sort atoms in a list of dictionaries, optionally using a reference
    dictionary (D_hat).

    Parameters
    ----------
    *list_D : multiple ndarray
        List of dictionaries to be sorted.
    D_hat : ndarray, optional
        Reference dictionary for sorting.

    Returns
    -------
    list_D_sorted : list of ndarray
        List of sorted dictionaries.

    """
    # If D_hat is provided, sort atoms in each dictionary in the list
    if D_ref is not None:
        return [sort_atoms(this_D, D_ref) for this_D in list_D]

    # If only one dictionary in the list, return it
    if len(list_D) == 1:
        return list_D

    # Recursive call to sort the rest of the dictionaries using the first as a reference
    return [list_D[0]] + sort_list_D(*list_D[1:], D_ref=list_D[0])


def sort_atoms(D, D_ref=None, return_permutation=False):
    """Sort the atoms in D_hat based on their correlation with the atoms in D.

    Parameters
    ----------
    D : ndarray of shape (n_atoms, n_channels, n_times_atom)
        The dictionary to be sorted.
    D_ref : ndarray of shape (n_atoms, n_channels, n_times_atom)
        The reference dictionary.

    Returns
    -------
    D_hat_sorted : ndarray of shape (n_atoms, n_channels, n_times_atom)
        The sorted version of D_hat. The atoms in D_hat_sorted correspond
        to the atoms in D in the sense that they have the maximum correlation.

    Notes
    -----
    The correlation between two atoms is computed by flattening their 2D
    representations into 1D arrays and computing the correlation coefficient.

    """
    if D_ref is None:
        if return_permutation:
            return D, None
        return D

    # Compute the correlation matrix
    n_atoms_ref, n_atoms = D_ref.shape[0], D.shape[0]
    corr_matrix = np.zeros((n_atoms_ref, n_atoms))
    for i in range(n_atoms_ref):
        for j in range(n_atoms):
            corr_matrix[i, j] = np.corrcoef(D_ref[i].flatten(), D[j].flatten())[0, 1]

    # Find the best match for each atom in D
    best_match = []
    for i in range(n_atoms_ref):
        for j in np.argsort(corr_matrix[i])[::-1]:
            if j not in best_match:
                best_match.append(j)
                break

    # handle atoms in D that were not matched
    used_atoms = best_match.copy()
    for j in range(n_atoms):
        if j not in used_atoms:
            best_match.append(j)

    # Sort the atoms in D
    D_sorted = D[best_match]

    if return_permutation:
        return D_sorted, best_match

    return D_sorted
