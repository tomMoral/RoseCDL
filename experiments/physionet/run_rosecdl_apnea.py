from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
from tqdm import tqdm
from utils_apnea import load_ecg

from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_outlier_comparison import remove_outliers_before_cdl

mem = Memory(location="__cache__", verbose=0)
EXP_DIR = Path("results") / "rosecdl_apnea"
EXP_DIR.mkdir(exist_ok=True, parents=True)


@mem.cache
def run_one(subject_id, cdl_params, outliers_kwargs, timing, i, seed, exp_dir, data_path=None):

    om = None if outliers_kwargs.get("method") == "none" else outliers_kwargs

    # Load the data
    if data_path:
        X, labels = load_ecg(subject_id, data_path=data_path, verbose=False)
    else:
        X, labels = load_ecg(subject_id, verbose=False)

    if timing == "before" and om:
        zshape = (X.shape[0], X.shape[1], X.shape[-1] - cdl_params["kernel_size"] + 1)
        X = remove_outliers_before_cdl(X, zshape, **outliers_kwargs)



    results = []

    def callback_fn(model, epoch, loss):
        # if outliers_kwargs is None:
        #     outliers_kwargs = {}

        results.append(
            {
                "run_idx": i,
                "subject_id": subject_id,
                "epoch": epoch,
                "loss": loss,
                "timing": timing,
                **outliers_kwargs,
            },
        )

    if timing == "during" and om:
        rosecdl = RoseCDL(
            **cdl_params,
            callbacks=[callback_fn],
            random_state=seed,
            outliers_kwargs=om,
        )
        rosecdl.fit(X)
    elif timing == "during" and not om:
        return []

    elif timing == "before" and om:
        rosecdl = RoseCDL(
            **cdl_params,
            callbacks=[callback_fn],
            random_state=seed,
        )
        rosecdl.fit(X)

    elif timing == "before" and not om:
        return []

    elif timing == "never":
        rosecdl = RoseCDL(
            **cdl_params,
            callbacks=[callback_fn],
            random_state=seed,
        )
        rosecdl.fit(X)


    # Plot the Dictionary
    D_hat=rosecdl.D_hat_

    fig, ax = plt.subplots(1, len(D_hat), figsize=(15, 3))
    for i, d_hat in enumerate(D_hat):
        ax[i].plot(d_hat.squeeze())

    if timing == "never":
        plt.savefig(exp_dir / f"atoms_{subject_id}_{i}_{timing}.pdf",
        bbox_inches="tight",
        format="pdf",
        )
    else:
        plt.savefig(exp_dir / f"atoms_{subject_id}_{i}_{timing}_{outliers_kwargs!s}.pdf",
        bbox_inches="tight",
        format="pdf",
        )

    plt.close()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RoseCDL on PhysioNet apnea ECG data",
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the PhysioNet apnea ECG data",
    )
    parser.add_argument(
        "--list-subjects",
        "-l",
        nargs="+",
        type=str,
        required=True,
        help="List of subject IDs to process",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=0.2,
        help="Regularization parameter",
    )
    parser.add_argument(
        "--n-jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel jobs",
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

    args = parser.parse_args()

    n_runs = args.n_runs
    reg = args.reg

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    cdl_configs = {
        "rosecdl": {
            "n_components": 3,
            "kernel_size": 70,
            "n_channels": 1,
            "lmbd": reg,
            "scale_lmbd": True,
            "epochs": 30,
            "max_batch": 10,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "linesearch",
            "n_iterations": 50,
            "window": True,
            "device": DEVICE,
        },
    }

    outlier_detection_methods = [
        {"method": "none"},
        {"method": "quantile", "alpha": 0.1},
        {"method": "quantile", "alpha": 0.2},
        # {"method": "iqr", "alpha": 1.5},
        # {"method": "zscore", "alpha": 1.5},
        {"method": "mad", "alpha": 3.5},
    ]
    outlier_detection_timings = ["before", "during", "never"]
    outliers_misc = {
        "moving_average": None,
        "union_channels": True,
        "opening_window": True,
    }

    for i in outlier_detection_methods:
        if i:
            i.update(outliers_misc)

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            subject_id,
            cdl_configs["rosecdl"],
            om,
            timing,
            i,
            args.seed,
            EXP_DIR,
            data_path=Path(args.data_path),
        )
        for i in range(n_runs)
        for subject_id in args.list_subjects
        for om in outlier_detection_methods
        for timing in outlier_detection_timings
    )

    results = [r for res in tqdm(results, "Running", total=n_runs) for r in res]

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(EXP_DIR / f"results_{args.reg}.csv", index=False)
    print(f"Results saved to {EXP_DIR / f'results_{args.reg}.csv'}")

    # Plot the losses with quantiles
    fig, ax = plt.subplots()
    for om, df_om in df.groupby("method"):
        # plot the median loss over epochs for each method
        median_loss = df_om.groupby("epoch")["loss"].median()
        q20 = df_om.groupby("epoch")["loss"].quantile(0.2)
        q80 = df_om.groupby("epoch")["loss"].quantile(0.8)

        # label = f"{om} (alpha={df_om['alpha'].iloc[0]})" if om != "none" else om

        ax.plot(median_loss, label=om)
        ax.fill_between(median_loss.index, q20, q80, alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(EXP_DIR / f"losses_{args.reg}.pdf", bbox_inches="tight", format="pdf")
    plt.show()
    plt.close()
