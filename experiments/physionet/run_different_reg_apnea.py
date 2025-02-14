"""Run experiments for the outliers detection task using the RoseCDL algorithm.

It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""  # noqa: INP001

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from scipy.signal.windows import tukey
from torch import cuda
from tqdm import tqdm
from wfdb.io.annotation import rdann
from wfdb.io.record import rdrecord

from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_outlier_comparison import remove_outliers_before_cdl

mem = Memory(location="__cache__", verbose=0)



def load_ecg(
    subject_id: str ="a01",
    T=60,
    data_path=Path("apnea-ecg"),
    apply_window=True,
    verbose=True,
):
    """Parameters
    ----------
    subject_id : str
    T : float
        duration, in seconds, of data splits
        default is 60 as the data have been annoted by the minute
    data_path : pathlib.Path
        path to data folder
    apply_window : bool (default: True)
        If set to True (default), a tukey window is applied to each split to
        reduce the border artifacts by reducing the weights of the chunk
        borders.
    verbose : bool
        if True, will print some information

    Returns
    -------
    X : ndarray, shape (n_splits, n_channels, int(T * fs))
        The signal splitted in ``n_splits``,
        whith ``n_splits = sig_len // int(T * fs)``,
        fs being the sampling frequency of the record
    labels : 1d array
        labels corresponding to one minute segments
        i.e., if T = 60, labels have the same length as data and each label
        corresponds to each datta split.

    """  # noqa: D205
    # ECG record
    record_name = str(data_path / subject_id)
    ecg_record = rdrecord(record_name=record_name)

    # split signal
    fs = ecg_record.fs  # sampling frequency of the record
    if verbose:
        print(f"Sampling frequency of the record: {fs} Hz")
    n_times = int(T * fs)
    n_splits = ecg_record.sig_len // n_times
    X = ecg_record.p_signal[: n_splits * n_times, :].T
    X = X.reshape(ecg_record.n_sig, n_splits, n_times).swapaxes(0, 1)

    # Apply a window to the signal to reduce the border artifacts
    if apply_window:
        X *= tukey(n_times, alpha=0.1)[None, None, :]

    # Add labels
    ann = rdann(
        record_name=record_name,
        extension="apn",
        return_label_elements=["symbol"],
        summarize_labels=True,
    )
    labels = np.array(ann.symbol)
    if T == 60:
        # ensure that labels and data have the same number of trials
        n_trials = min(len(ann.symbol), n_splits)
        return X[:n_trials], labels[:n_trials]
    import warnings

    warnings.warn(
        f"The returned labels do not match the data as T != 60 (got T = {T}).",
    )
    return X, labels

def generate_run_config_list(  # noqa: PLR0913
    cdl_packages: list[str],
    regularization_values: list[float],
    outlier_detection_timings: list[str],
    cdl_configs: dict[str, dict[str, str | float | int | bool]],
    outlier_methods: list[dict[str, str | float]],
    n_runs: int = 1,
    seed: int | None = None,
):
    """Generate the list of configurations for the experiment.

    Args:
        cdl_packages (list): List of CDL packages to use.
        regularization_values (list): List of regularization values to test.
        outlier_detection_timings (list): List of outlier detection timings.
        cdl_configs (dict): Base CDL configurations.
        outlier_methods (list): List of outlier detection method configurations.
            Each method should be a dict with keys 'method' and 'alpha'.
        n_runs (int): Number of runs to generate.
        seed (int): Master seed for the experiment.
    """
    rng = np.random.default_rng(seed)
    list_seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    run_config_list = []
    for package in cdl_packages:
        for reg in regularization_values:
            package_config = cdl_configs[package].copy()
            package_config["lmbd"] = reg

            for timing in outlier_detection_timings:
                if package != "rosecdl" and timing == "during":
                    continue

                if timing == "never":
                    methods_to_test = [{"method": "none", "alpha": -1}]
                else:
                    methods_to_test = outlier_methods

                for method in methods_to_test:
                    run_config_list.extend(
                        {
                            "cdl_package": package,
                            "cdl_params": package_config,
                            "outlier_detection_method": method,
                            "outlier_detection_timing": timing,
                            "seed": s,
                            "i": i,
                        }
                        for i, s in enumerate(list_seeds)
                    )
    return run_config_list


@mem.cache
def run_one(
    cdl_package: str,
    cdl_params: dict[str, str or float],
    outlier_detection_method: dict[str, str or float],
    outlier_detection_timing: str,
    seed: int,
    i: int,
    outliers_kwargs: dict[str, str or float],
    data_path: Path|str,
    exp_dir: str,
):
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        cdl_package (str): CDL package name: "rosecdl", "alphacsc" or "sporco".
        outlier_detection_method (str): Outlier detection method.
            A dictionary is expected with two keys:
                - "name": name of the method. Can be one of "quantile", "iqr",
                    "zscore", "mad" or "none".
                - "alpha" (float): Parameter of the method.
        outlier_detection_timing (str): When to run the outlier detection
            relatively to the CDL: "before", "during" or "never".
        outliers_kwargs (dict): Additional parameters for outlier detection.
        cdl_params (dict): Parameters for the CDL algorithm.
        seed (int): Random seed.
        i (int): Counting index of the run.
        exp_dir (str): Name of the directory to store the results

    """
    # Process the outlier method"s parameters and compute the summary name
    if outlier_detection_timing == "never":
        summary_name, outliers_kwargs = "no detection", None
        outlier_detection_method = {}
    else:
        summary_name = "{method} (alpha={alpha:.02f})".format(
            **outlier_detection_method,
        )
        outliers_kwargs = {**outliers_kwargs, **outlier_detection_method}

    summary_name = f"[{cdl_package}] {summary_name}"
    if outlier_detection_timing != "never":
        summary_name += f" ({outlier_detection_timing})"

    print(f"Running {summary_name} reg = {cdl_params['lmbd']} ({seed})")

    if cdl_package == "sporco":
        return []

    if isinstance(data_path, str):
        data_path = Path(data_path)

    # Load the data
    data, labels = load_ecg(subject_id="a02", data_path=data_path)

    Xbad = data[labels == "A"]
    Xgood = data[labels == "N"]

    X = Xbad[:10]


    # Perform outlier detection on the data before CDL
    # XXX: Get z shape otherwise :
    # zshape = (X.shape[0], X.shape[1], X.shape[-2]-kernel_size[0]+1, X.shape[-1]-kernel_size[1]+1)

    if outlier_detection_timing == "before":
        zshape = (
            X.shape[0],
            X.shape[1],
            X.shape[-1] - cdl_params["kernel_size"] + 1,
        )

        X = remove_outliers_before_cdl(
            data=X,
            activation_vector_shape=zshape,
            **outliers_kwargs,
        )
        outliers_kwargs = None

    # Setup the callback
    results = []

    # Run the experiment
    if cdl_package == "rosecdl":
        rng = np.random.default_rng(seed)
        D_init = rng.normal(
            size=(cdl_params["n_components"],
            cdl_params["n_channels"],
            cdl_params["kernel_size"]),
        )

        cdl = RoseCDL(
            **cdl_params,
            D_init=D_init,
            outliers_kwargs=outliers_kwargs,
        )
        cdl.fit(X)

        if i == 0:
            # Plot result dictionary
            fig, axes = plt.subplots(1, cdl.D_hat_.shape[0], figsize=(12, 3))
            for i, ax in enumerate(axes):
                ax.plot(cdl.D_hat_[i].squeeze())
                ax.set_yticks(np.linspace(-0.5, 0.5, 3))
            plt.suptitle(f"RoseCDL - {summary_name} - reg = {cdl_params['lmbd']}")
            fig.savefig(
                exp_dir / f"D_hat_{summary_name.replace(' ', '_')}_reg_{cdl_params["lmbd"]}.pdf"
                )

    else:
        raise ValueError(f"Unknown CDL package {cdl_package}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a comparison of the different methods for dictionary recovery in 2D"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to NPZ file containing the data (X and d arrays)",
    )
    parser.add_argument(
        "--n-jobs", "-j", type=int, default=1, help="Number of parallel jobs",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Master seed for reproducible job",
    )

    # parser.add_argument(
    #     "--n-runs",
    #     "-n",
    #     type=int,
    #     default=20,
    #     help="Number of repetitions for the experiment",
    # )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory to store the results",
    )
    args = parser.parse_args()

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")  # noqa: T201

    seed = args.seed
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**32 - 1)
    print(f"Seed: {seed}")  # noqa: T201

    # Replace the dynamic n_runs with a fixed value since only the dictionary matters
    n_runs = 1

    exp_dir = Path(args.output) / "reg_impact_dict_quality_apnea"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Base contamination parameters
    contamination_params = {}

    # Define base CDL parameters
    cdl_packages = ["rosecdl"]
    cdl_configs = {
        "rosecdl": {
            "kernel_size": 100,
            "n_channels": 1,
            "n_components": 3,
            "scale_lmbd": True,
            "epochs": 50,
            "max_batch": 20,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "adam",
            "n_iterations": 60,
            "window": True,
            "device": DEVICE,
            "positive_D": False,
        },
        "sporco": {},
    }

    regularization_values = [0.01, 0.1, 0.3, 0.5, 0.8]

    # Fixed outlier detection parameters
    outlier_detection_timings = ["before", "during", "never"]
    outliers_kwargs = {
        "moving_average": None,
        "union_channels": True,
        "opening_window": True,
    }

    # Define multiple outlier detection methods to test
    outlier_methods = [
        {"method": "mad", "alpha": 3.5},
        {"method": "zscore", "alpha": 1.0},
        {"method": "iqr", "alpha": 1.5},
        {"method": "quantile", "alpha": 0.05},
        {"method": "quantile", "alpha": 0.1},
        {"method": "quantile", "alpha": 0.2},
    ]

    run_configs = generate_run_config_list(
        cdl_packages=cdl_packages,
        regularization_values=regularization_values,
        outlier_detection_timings=outlier_detection_timings,
        cdl_configs=cdl_configs,
        outlier_methods=outlier_methods,
        n_runs=n_runs,
        seed=seed,
    )

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            **run_config,
            outliers_kwargs=outliers_kwargs,
            data_path=args.data_path,
            exp_dir=exp_dir,
        )
        for run_config in run_configs
    )
    results = [
        r for res in tqdm(results, "Running", total=len(run_configs)) for r in res
    ]
