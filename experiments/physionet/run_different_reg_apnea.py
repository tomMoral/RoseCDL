"""Run experiments for the outliers detection task using the RoseCDL algorithm.

It saves the results in a csv file that will be used by
plot_outliers_detection.py to generate the plots.
"""  # noqa: INP001

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alphacsc import BatchCDL
from alphacsc.init_dict import init_dictionary
from alphacsc.loss_and_gradient import compute_objective
from alphacsc.update_z_multi import update_z_multi
from alphacsc.utils.convolution import construct_X_multi
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

            if package == "rosecdl":
                package_config["lmbd"] = reg
            elif package == "alphacsc":
                package_config["reg"] = reg
            else:
                msg = f"Unknown CDL package {package}"
                raise ValueError(msg)

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

    if cdl_package == "rosecdl":
        print(f"Running {summary_name} reg = {cdl_params['lmbd']} ({seed})")
    elif cdl_package == "alphacsc":
        print(f"Running {summary_name} reg = {cdl_params['reg']} ({seed})")
    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

    if cdl_package == "sporco":
        return []

    if isinstance(data_path, str):
        data_path = Path(data_path)

    # Load the data
    data, labels = load_ecg(subject_id="a02", data_path=data_path)

    Xbad = data[labels == "A"]
    Xgood = data[labels == "N"]

    X = Xbad[:10]


    if cdl_package == "rosecdl":
        kernel_size = cdl_params["kernel_size"]
    elif cdl_package == "alphacsc":
        kernel_size = cdl_params["n_times_atom"]


    if outlier_detection_timing == "before":
        zshape = (
            X.shape[0],
            X.shape[1],
            X.shape[-1] - kernel_size + 1,
        )

        X = remove_outliers_before_cdl(
            data=X,
            activation_vector_shape=zshape,
            **outliers_kwargs,
        )
        outliers_kwargs = None

    assert X.ndim == 3, f"X should have 3 dimensions, got {X.ndim}"

    # Setup the callback
    results = []

    def callback_fn(model, *args) -> None:
        if len(args) == 1:
            pobj = args[0]
            # alphacsc adds the loss twice in pobj per epoch,
            # Divide by two the epoch number
            loss, epoch = 0 if len(pobj) == 0 else pobj[-1], len(pobj) // 2
            print(f"Epoch {epoch} - Loss {loss}")
        else:
            epoch, loss = args

        if epoch % 2 == 0:
            return

        D_hat = model.D_hat_ if cdl_package == "rosecdl" else model.D_hat

        regu = cdl_params["lmbd"] if cdl_package == "rosecdl" else cdl_params["reg"]

        z_hat, _, _ = update_z_multi(Xgood[:10], D_hat, reg=regu)
        X_hat = construct_X_multi(z_hat, D=D_hat)
        cost = compute_objective(
            Xgood[:10],
            X_hat=X_hat,
            z_hat=z_hat,
            D=D_hat,
            reg=regu,
        )

        results.append(
            {
                "name": summary_name,
                **outlier_detection_method,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "cost": cost,
                "method": cdl_package,
                "lmbd": regu,
            },
        )

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
            callbacks=[callback_fn],
        )
        cdl.fit(X)

        if i == 0:
            # Plot result dictionary
            fig, axes = plt.subplots(1, cdl.D_hat_.shape[0], figsize=(12, 3))
            for idx, ax in enumerate(axes):
                ax.plot(cdl.D_hat_[idx].squeeze())
                ax.set_yticks(np.linspace(-0.5, 0.5, 3))
            plt.suptitle(f"RoseCDL - {summary_name} - reg = {cdl_params['lmbd']}")
            fig.savefig(
                exp_dir / f"D_hat_{summary_name.replace(' ', '_')}_reg_{cdl_params["lmbd"]}.pdf"
                )
    elif cdl_package == "alphacsc":
        rng = np.random.default_rng(seed)
        D_init = init_dictionary(
            X,
            cdl_params["n_atoms"],
            cdl_params["n_times_atom"],
            window=True,
            rank1=cdl_params["rank1"],
            D_init="random",
            random_state=seed,
        )

        cdl = BatchCDL(**cdl_params, D_init=D_init)
        cdl.raise_on_increase = False
        cdl.callback = callback_fn
        cdl.fit(X)

        if i == 0:
            # Plot result dictionary
            fig, axes = plt.subplots(1, cdl.D_hat.shape[0], figsize=(12, 3))
            for idx, ax in enumerate(axes):
                ax.plot(cdl.D_hat[idx].squeeze())
                ax.set_yticks(np.linspace(-0.5, 0.5, 3))
            plt.suptitle(f"RoseCDL - {summary_name} - reg = {cdl_params['reg']}")
            fig.savefig(
                exp_dir / f"D_hat_{summary_name.replace(' ', '_')}_reg_{cdl_params["reg"]}.pdf"
                )

    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the outlier detection experiment for the apnea dataset",
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
    parser.add_argument(
        "--n-runs",
        "-n",
        type=int,
        default=20,
        help="Number of repetitions for the experiment",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory to store the results",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode",
    )
    args = parser.parse_args()

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")  # noqa: T201

    seed = args.seed
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**32 - 1)
    print(f"Seed: {seed}")  # noqa: T201

    n_runs = args.n_runs

    # Fix t nruns and jobs to 1 for debugging
    if args.debug:
        n_runs = 1
        args.n_jobs = 1

    exp_dir = Path(args.output) / "reg_impact_dict_quality_apnea"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Define base CDL parameters
    cdl_packages = ["rosecdl", "alphacsc"]
    # cdl_packages = ["alphacsc"]
    cdl_configs = {
        "rosecdl": {
            "kernel_size": 100,
            "n_channels": 1,
            "n_components": 3,
            "scale_lmbd": True,
            "epochs": 10 if args.debug else 30,
            "max_batch": 20,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "adam",
            "n_iterations": 10 if args.debug else 40,
            "window": True,
            "device": DEVICE,
            "positive_D": False,
        },
        "alphacsc": {
            "n_times_atom": 100,
            "n_atoms": 3,
            "lmbd_max": "scaled",
            "n_iter": 10 if args.debug else 30,
            "solver_z": "lgcd",
            "rank1": False,
            "window": True,
            "verbose": 0,
        },
    }

    regularization_values = [0.01] if args.debug else [0.01, 0.1, 0.3, 0.5, 0.8]

    # Fixed outlier detection parameters
    outlier_detection_timings = ["before"] if args.debug else ["before", "during", "never"]
    outliers_kwargs = {
        "moving_average": None,
        "union_channels": True,
        "opening_window": True,
    }

    # Define multiple outlier detection methods to test
    outlier_methods = [
        {"method": "mad", "alpha": 3.5},
    ] if args.debug else [
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

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    # Plot recovery score with separate curves for each reg value
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Group by name, regularization value, and epoch
    curves = (
        df_results.groupby(["name", "lmbd", "epoch"])["cost"]
        .quantile([0.2, 0.5, 0.8])
        .unstack()
    )

    # Plot each combination of name and reg value
    color_idx = 0
    for name in df_results.name.unique():
        reg_values = df_results[df_results.name == name]["lmbd"].unique()
        for reg in reg_values:
            label = f"{name} (reg={reg})"
            idx = (name, reg)
            if idx in curves.index:
                ax.fill_between(
                    curves.loc[idx].index,
                    curves.loc[idx, 0.2],
                    curves.loc[idx, 0.8],
                    alpha=0.3,
                    color=f"C{color_idx}",
                    label=None,
                )
                curves.loc[idx, 0.5].plot(label=label, c=f"C{color_idx}")
                color_idx += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Full cost")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(exp_dir / "full_cost_apnea.pdf", bbox_inches="tight")
