import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
from sporco.dictlrn import cbpdndl
from torch.linalg import vector_norm
from torch.nn import MSELoss
from tqdm import tqdm

from rosecdl.loss import LassoLoss, _ReconstructionLoss
from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat, fista

mem = Memory(location="__cache__", verbose=0)

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_run_config_list(
    cdl_packages: list[str],
    cdl_configs: dict[str, dict],
    n_runs: int = 1,
    seed: int | None = None,
) -> list[dict[str, any]]:
    """Generate the list of configurations for the experiment.

    Args:
        cdl_packages (list): List of CDL packages to use.
        cdl_configs (dict): Dictionary of CDL configurations.
        n_runs (int): Number of runs to generate.
        seed (int): Master seed for the experiment.

    """
    # Generate a list of seeds for reproducibility
    rng = np.random.default_rng(seed)
    list_seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    run_config_list = []
    for package in cdl_packages:
        run_config_list.extend(
            {
                "cdl_package": package,
                "cdl_params": cdl_configs[package],
                "seed": s,
                "i": i,
            }
            for i, s in enumerate(list_seeds)
        )
    return run_config_list


@mem.cache
def run_one(
    data_path: str,
    cdl_package: str,
    cdl_params: dict[str, str or float],
    seed: int,
    i: int,
) -> list:
    """Run the experiment for a given CDL package and outlier detection method.

    Args:
        data_path (str): Path to the data file.
        cdl_package (str): CDL package name: "rosecdl", "alphacsc" or "sporco".
        cdl_params (dict): Parameters for the CDL algorithm.
        seed (int): Random seed.
        i (int): Counting index of the run.

    """
    logger.info(
        "Running %s with seed %d (run %d)",
        cdl_package,
        seed,
        i,
    )

    data_params = np.load(data_path)
    data = data_params["X"]
    true_dict = data_params["D"]
    true_dict = true_dict[:, None, ...]

    # test_data = data_params["X_test"]

    if data.ndim == 2:
        data = data[None, None, ...]

    s1 = 990
    s2 = 550
    t1 = s1 + 900
    t2 = s2 + 900
    data = data[:, :, s1:t1, s2:t2]

    plt.imshow(data[0, 0], cmap="gray")
    plt.title("Data")
    plt.savefig(exp_dir / f"data_{i}.png")
    plt.close()

    # rng = np.random.default_rng(seed)
    # init_dict = rng.standard_normal(
    #     (
    #         cdl_params["n_components"],
    #         cdl_params["n_channels"],
    #         *cdl_params["kernel_size"],
    #     )
    # )

    init_dict = np.random.default_rng(seed).standard_normal((6, 1, 35, 30))
    data = torch.tensor(data, device=cdl_params["device"])

    cdl_params = cdl_params.copy()
    if cdl_package in ["rosecdl", "deepcdl"]:
        init_dict = torch.tensor(init_dict, device=cdl_params["device"])
        data = torch.tensor(data, device=cdl_params["device"])
        # cdl_params["lmbd"] *= lmbd_max
        lmbd = cdl_params["lmbd"]
    elif cdl_package == "sporco":
        # cdl_params["lmbda"] *= lmbd_max
        lmbd = cdl_params["lmbda"]
    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

    # Setup the callback
    results, t_start = [], time.perf_counter()
    z0_dict = {"train": None, "test": None}

    # Instantiating a new RoseCDL object for computing the loss
    # Alphacsc compute_objective alternative for 2D
    # evaluation_model = RoseCDL(
    #     lmbd=lmbd,
    #     D_init=init_dict
    #     if cdl_package in ["rosecdl", "deepcdl"]
    #     else torch.tensor(init_dict, device=cdl_params["device"]),
    #     window=False,
    #     outliers_kwargs=None,
    #     device=cdl_params["device"],
    #     n_iterations=5,
    # )

    def callback_fn(model, *args) -> None:
        nonlocal t_start
        runtime = time.perf_counter() - t_start

        if cdl_package == "sporco":
            # Sporco returns the dimension of the dictionary in a different order
            model_dict = model.getdict()[:, :, 0, :, :].transpose(3, 2, 1, 0)
            # model_dict /= model_dict.norm(dim=(1, 2, 3), keepdim=True)
            model_dict = torch.tensor(model_dict, device=cdl_params["device"])
            epoch, loss = len(results), None

        else:
            epoch, loss = args
            model_dict = model.D_hat_
            model_dict = torch.tensor(model_dict, device=cdl_params["device"])

        fig, ax = plt.subplots(1, len(model_dict), figsize=(15, 5))
        for l, atom in enumerate(model_dict):
            ax[l].imshow(atom[0].cpu().numpy(), cmap="gray")
            ax[l].set_title(f"Atom {i + 1}")
            ax[l].axis("off")
        plt.savefig(exp_dir / f"dict_{epoch}.png")
        plt.close(fig)

        # xh, zh = evaluation_model.csc(
        #     x=eval_data,
        #     D=model_dict,
        # )

        zh = fista(
            data.clone(),
            model_dict,
            lmbd,
            zO=None,
            n_iter=150,
        )
        z0_dict["train"] = zh.clone()

        xh = torch.nn.functional.conv_transpose2d(zh.clone(), model_dict.clone())

        logger.info(
            "sum x = %s, sum zh = %s",
            torch.sum(xh),
            torch.sum(zh),
        )



        # plot xh and zh
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(xh[0, 0].cpu().numpy(), cmap="gray")
        plt.title("Reconstructed image")
        plt.subplot(1, 2, 2)
        plt.imshow(zh[0, 0].cpu().numpy(), cmap="gray")
        plt.title("Sparse code")
        plt.savefig(exp_dir / f"reconstructed_{epoch}.png")

        # loss_true = evaluation_model.loss_fn(
        #     xh,
        #     zh,
        #     eval_data,
        # )

        loss_true = LassoLoss(
            lmbd=cdl_params["lmbd"],
            reduction="mean",
            data_fit=MSELoss(reduction="mean"),
        )(
            xh,
            zh,
            data,
        )
        loss_true = loss_true.item()

        logger.info(
            "Epoch %d, loss: %.4f, loss_true: %.4f, time: %.2fs",
            epoch,
            loss,
            loss_true,
            runtime,
        )

        recovery_score = evaluate_D_hat(true_dict, model_dict.cpu().numpy())
        results.append(
            {
                "name": cdl_package,
                "recovery_score": recovery_score,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "loss_true": loss_true,
                "test_loss_true": -1,
                "time": runtime,
            }
        )
        t_start = time.perf_counter()

    # Run the experiment
    if cdl_package in ["rosecdl", "deepcdl"]:
        if not isinstance(init_dict, torch.Tensor):
            init_dict = torch.tensor(init_dict, device=cdl_params["device"])
        cdl = RoseCDL(
            **cdl_params,
            D_init=init_dict,
            callbacks=[callback_fn],
        )
        # Setting up classical conv instead of fftconv for deepcdl
        if cdl_package == "deepcdl":
            cdl.csc.conv_algo = "classical"
            cdl.csc.set_conv_methods()
        cdl.fit(data)
    elif cdl_package == "sporco":
        opt_cbpdn = cbpdndl.ConvBPDNOptionsDefaults()

        opt = cbpdndl.ConvBPDNDictLearn.Options(
            {
                "Verbose": False,
                "MaxMainIter": cdl_params["n_iter"],
                "CBPDN": opt_cbpdn,
                "Callback": callback_fn,
            },
            dmethod="cns",
        )

        if isinstance(init_dict, torch.Tensor):
            init_dict = init_dict.cpu().numpy()
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        sporco_params = {
            "D0": init_dict.transpose(3, 2, 1, 0).copy(),
            "S": data.transpose(3, 2, 1, 0).copy(),
            "lmbda": cdl_params["lmbda"],
            "opt": opt,
            "dmethod": "cns",
            "dimN": 2,
        }

        cdl = cbpdndl.ConvBPDNDictLearn(**sporco_params)
        cdl.solve()
    else:
        msg = f"Unknown CDL package {cdl_package}"
        raise ValueError(msg)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a comparison of the different methods for dictionary recovery"
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        required=True,
        help="Path to the data file",
    )
    parser.add_argument(
        "--n-jobs", "-j", type=int, default=1, help="Number of parallel jobs"
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
        "--reg", type=float, default=0.8, help="Regularization parameter"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the script in debug mode"
    )
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", DEVICE)

    seed = args.seed
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**32 - 1)
    logger.info("Seed: %s", seed)

    n_runs = 1 if args.debug else args.n_runs
    reg = args.reg

    data_path = args.data_path

    exp_dir = Path(args.output) / f"2D_runtime_comparison_{reg}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Define base CDL parameters
    # cdl_packages = ["deepcdl", "rosecdl", "sporco"]
    cdl_packages = ["rosecdl"]
    cdl_configs = {
        "rosecdl": {
            "kernel_size": (35, 30),
            "n_channels": 1,
            "n_components": 6,
            "lmbd": reg,
            "scale_lmbd": True,
            "epochs": 5 if args.debug else 30,
            "max_batch": None,
            "mini_batch_size": 10,
            "sample_window": 1000,
            "optimizer": "linesearch",
            "n_iterations": 5 if args.debug else 50,
            "window": True,
            "device": DEVICE,
        },
        "sporco": {
            "lmbda": reg,
            "n_iter": 5 if args.debug else 30,
            "device": DEVICE,
        },
    }
    cdl_configs["deepcdl"] = cdl_configs["rosecdl"].copy()
    cdl_configs["deepcdl"]["deepcdl"] = True

    run_configs = generate_run_config_list(
        cdl_packages=cdl_packages,
        cdl_configs=cdl_configs,
        n_runs=n_runs,
        seed=seed,
    )

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(**run_config, data_path=data_path)
        for run_config in run_configs
    )
    results = [
        r for res in tqdm(results, "Running", total=len(run_configs)) for r in res
    ]

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "df_results.csv", index=False)

    # Plotting train loss for the different methods
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["loss_true"] = (
            median_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["loss_true"] = (
            q02_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        q8_curve = name_data[["time", "loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["loss_true"] = (
            q8_curve["loss_true"] - df_results["loss_true"].min() + 1e1
        )

        (line,) = ax.plot(median_curve["time"], median_curve["loss_true"], label=name)
        color = line.get_color()
        ax.fill_between(
            median_curve["time"],
            q02_curve["loss_true"],
            q8_curve["loss_true"],
            color=color,
            alpha=0.2,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Train Loss")
    plt.title("Speed of convergence of the different methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "loss_true.png")
    plt.show()
