import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from .utils.utils_outliers import compute_error, get_outlier_mask


def get_lambda_max(X, D_hat, sample_weights=None, q=1, method="quantile"):
    """For each atom, compute the regularization parameter scaling.
    This value is usually defined as the smallest value for which 0 is
    a solution of the optimization problem.
    In order to avoid spurious values, this quantity can also be estimated
    as the q-quantile of the correlation between signal patches and the
    atom.

    Parameters
    ----------
    X : array, shape (n_trials, n_times) or
               shape (n_trials, n_channels, n_times)
        The data

    D_hat : array, shape (n_atoms, n_channels + n_times_atoms) or
                   shape (n_atoms, n_channels, n_times_atom)
        The atoms

    sample_weights : None | array, shape (n_trials, n_times) or
                                   shape (n_trials, n_channels, n_times)
        Weights to apply to the data.
        Defaults is None

    q : float
        If method is quantile (default), the quantile to compute, which must be between 0 and 1 inclusive.
        Default is 1, i.e., the maximum is returned.
        If method is iqr, zscore or mad, the alpha parameter to use.

    method : str
        Method to use to compute the lambda_max. Should be one of:
        - quantile: q-quantile of the correlation between signal patches and the
                    atom
        - iqr: interquartile range of the correlation between signal patches
               and the atom
               Recommended value for alpha: 1.5
        - zscore: mean + alpha * std of the correlation between signal patches
                  and the atom
                  Recommended values for alpha: 1, 2 or 3
        - mad: median + alpha * mad of the correlation between signal patches
               and the atom
               Recommended values for alpha: 3.5

    Returns
    -------
    lambda_max : array, shape (n_atoms, 1)
    """
    # univariate case, add a dimension (n_channels = 1)
    if X.ndim == 2:
        X = X[:, None, :]
        D_hat = D_hat[:, None, :]
        if sample_weights is not None:
            sample_weights = sample_weights[:, None, :]

    n_trials, n_channels, n_times = X.shape

    if sample_weights is None:
        # no need for the last dimension if we only have ones
        if D_hat.ndim == 2:
            sample_weights = np.ones(n_trials)
        else:
            sample_weights = np.ones((n_trials, n_channels))

    # multivariate rank-1 case
    if D_hat.ndim == 2:
        conv_res = [
            [
                np.convolve(
                    np.dot(uv_k[:n_channels], X_i * W_i),
                    uv_k[: n_channels - 1 : -1],
                    mode="valid",
                )
                for X_i, W_i in zip(X, sample_weights)
            ]
            for uv_k in D_hat
        ]
    # multivariate general case
    else:
        conv_res = [
            [
                np.sum(
                    [
                        np.correlate(D_kp, X_ip * W_ip, mode="valid")
                        for D_kp, X_ip, W_ip in zip(D_k, X_i, W_i)
                    ],
                    axis=0,
                )
                for X_i, W_i in zip(X, sample_weights)
            ]
            for D_k in D_hat
        ]

    if method == "quantile":
        assert q <= 1 and q >= 0
        return np.quantile(conv_res, axis=(1, 2), q=q)[:, None]
    elif method == "iqr":
        alpha = q
        assert alpha >= 1
        q1, q3 = np.quantile(conv_res, axis=(1, 2), q=[0.25, 0.75])
        res = q3 + 1.5 * (q3 - q1)
        return res[:, None]
    elif method == "zscore":
        alpha = q
        assert alpha >= 1
        res = np.mean(conv_res, axis=(1, 2)) + alpha * np.std(conv_res, axis=(1, 2))
        return res[:, None]
    elif method == "mad":
        alpha = q
        assert alpha >= 1
        median = np.median(conv_res, axis=(1, 2))
        mad = np.median(np.abs(conv_res - median[:, None, None]), axis=(1, 2))
        constant = 0.6745
        res = median + alpha * mad / constant
        return res[:, None]
    else:
        raise ValueError(
            f"Unknown method {method}, should be one of " f"quantile, iqr, zscore, mad"
        )


def multi_channel_2d_correlate(dk, pat):
    """
    Compute the multi-channel 2D correlation between a dictionary element and a pattern.

    Parameters:
        dk, pat : np.ndarray
            2D array of shape (n_channels, n_times_atom).

    Returns:
        np.ndarray : The 2D correlation.
    """
    return np.sum(
        [signal.correlate(dk_c, pat_c, mode="full") for dk_c, pat_c in zip(dk, pat)],
        axis=0,
    )


def compute_best_assignment(corr):
    """
    Compute the best assignment for the correlation matrix using the Hungarian algorithm.

    Parameters:
        corr : np.ndarray
            The correlation matrix.

    Returns:
        float : The mean value of the best assignment.
    """
    i, j = linear_sum_assignment(corr, maximize=True)
    return corr[i, j].mean()


def evaluate_D_hat(patterns, D_hat):
    """
    Evaluate the learned dictionary D_hat with respect to a set of patterns.

    Parameters:
        patterns : np.ndarray
            The set of patterns, a 3D array (n_patterns, n_channels, n_times_atom).
        D_hat : np.ndarray
            The learned dictionary, a 3D array (n_atoms, n_channels, n_times_atom).

    Returns:
        float : The evaluation score.
    """
    patterns, D_hat = patterns.copy(), D_hat.copy()

    # axis = (2, 3)
    axis = (1, 2)
    patterns /= np.linalg.norm(patterns, ord="f", axis=axis, keepdims=True)
    D_hat /= np.linalg.norm(D_hat, ord="f", axis=axis, keepdims=True)

    corr = np.array(
        [
            [multi_channel_2d_correlate(dk, pat).max() for dk in D_hat]
            for pat in patterns
        ]
    )
    return compute_best_assignment(corr)


def check_and_load_data(exp_dir, file_name):
    file_path = Path(exp_dir) / file_name
    if file_path.exists():
        return np.load(file_path)
    return None


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def check_and_initialize(
    exp_dir, simulation_params, wincdl_params, results_file_name="df_results"
):
    """
    Checks if previous simulation parameters exist and initializes the environment accordingly.

    Parameters:
    - exp_dir (Path): Directory where experiment results are stored.
    - simulation_params (dict): Current simulation parameters.
    - wincdl_params (dict): Current wincdl parameters.
    - results_file_name (str): Name of the CSV file to store results.

    Returns:
    - df_results (DataFrame): DataFrame to hold results, either loaded from file or initialized as empty.
    """

    # Define paths based on exp_dir
    simulation_path = exp_dir / "simulation_params.json"
    wincdl_path = exp_dir / "wincdl_params.json"
    results_path = exp_dir / f"{results_file_name}.csv"

    # Check if parameter files exist and compare them with current parameters
    if simulation_path.exists() and wincdl_path.exists():
        previous_simulation_params = load_json(simulation_path)
        previous_wincdl_params = load_json(wincdl_path)
        if (
            simulation_params != previous_simulation_params
            or wincdl_params != previous_wincdl_params
        ):
            print("Parameters changed, starting from scratch")
            df_results = pd.DataFrame()
            df_results.to_csv(results_path, index=False)
            save_json(simulation_params, simulation_path)
            save_json(wincdl_params, wincdl_path)
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
        save_json(wincdl_params, wincdl_path)

    return df_results


def get_method_name(outliers_kwargs):
    if outliers_kwargs is None:
        return "no detection"

    method = outliers_kwargs["method"]
    if "unilateral" in method:
        method = method.split("_")[0]
    if "alpha" in outliers_kwargs:
        alpha = outliers_kwargs["alpha"]
        method += f" (alpha={alpha})"

    return method


def get_outliers_metric(
    wincdl,
    true_outliers_mask,
    X,
    D,
    return_mask=False,
    per_patch=False,
    with_opening_window=True,
):
    """

    wincdl: WinCDL instance
    """
    X_hat = wincdl.get_prediction(X, D=D)

    kwargs = wincdl.outliers_kwargs
    moving_average = kwargs.pop("moving_average", None)
    opening_window = kwargs.pop("opening_window", True)
    if not with_opening_window:
        opening_window = False

    model = wincdl.csc
    err = compute_error(
        prediction=X_hat,
        X=X,
        loss_fn=wincdl.loss_fn,
        per_patch=model.n_times_atom if per_patch else False,
        device=model.device,
        z_hat=model._z_hat.clone() if per_patch else None,
        lmbd=model.lmbd,
    )
    # Compute outlier mask
    # thresholds = get_thresholds(
    #     err, method=outliers_kwargs['method'], alpha=outliers_kwargs['alpha']
    # )
    outliers_mask = (
        get_outlier_mask(
            data=err,
            thresholds=None,  # Recompute thresholds
            moving_average=moving_average if not per_patch else None,
            opening_window=model.n_times_atom if opening_window else None,
            **kwargs,
        )
        .detach()
        .cpu()
        .numpy()
    )

    if outliers_mask.ndim == 2 and X.ndim == 3:
        # Duplicate across channels
        outliers_mask = np.repeat(outliers_mask[:, None, :], X.shape[1], axis=1)

    # outliers_mask = get_outlier_mask(
    #     data=compute_error(
    #         X_hat,
    #         X,
    #         wincdl.loss_fn,
    #         per_patch=model.n_times_atom if PER_PATCH else False,
    #         keep_dim=True,
    #     ),
    #     thresholds=None,
    #     opening_window=wincdl.csc.n_times_atom if opening_window else None,
    #     **kwargs,
    # ).cpu().numpy()

    if wincdl.outliers_kwargs["union_channels"]:
        true_outliers_mask = np.minimum(true_outliers_mask.sum(axis=1), 1)

    # Ensure masks have the same shape
    if true_outliers_mask.shape != outliers_mask.shape:
        raise ValueError(
            f"Shape of true_outliers_mask ({true_outliers_mask.shape}) "
            f"does not match shape of outliers_mask ({outliers_mask.shape})"
        )

    accuracy = np.mean(outliers_mask == true_outliers_mask)
    precision = precision_score(true_outliers_mask.flatten(), outliers_mask.flatten())
    recall = recall_score(true_outliers_mask.flatten(), outliers_mask.flatten())
    f1 = f1_score(true_outliers_mask.flatten(), outliers_mask.flatten())
    # Compute dice score
    dice = 2 * (precision * recall) / (precision + recall)
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

    if return_mask:
        return score_dict, outliers_mask

    return score_dict


def plot_dicts(*dicts, D_true=None, labels=None, sup_title=None, sort_dicts=True):
    """
    Plot one or more dictionaries, with the option of overlaying a ground truth dictionary.

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
                    D_true[i, j, :], color="black", linestyle="--", label="D_true"
                )
            for d, label in zip(dicts, labels):
                if i == 0 and j == (n_channels - 1):
                    # Only add legend for top right subplot
                    label = label
                else:
                    label = None
                axs[i, j].plot(d[i, j, :], label=label, alpha=0.7)

            axs[i, j].set_title(f"Atom {i+1}, Channel {j+1}")
            if i == n_atoms - 1:
                axs[i, j].set_xlabel("Time")
            if j == n_channels - 1:
                axs[i, j].legend()

    plt.xlim(0, n_times_atom - 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if sup_title:
        fig.suptitle(sup_title)

    return fig


def sort_list_D(*list_D, D_ref=None):
    """
    Sort atoms in a list of dictionaries, optionally using a reference
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
    """
    Sort the atoms in D_hat based on their correlation with the atoms in D.

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
