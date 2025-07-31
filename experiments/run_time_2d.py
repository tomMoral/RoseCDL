import logging
import time
from pathlib import Path

# seed 3398290419
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Memory, Parallel, delayed
from sporco.dictlrn import cbpdndl
from torch import cuda
from tqdm import tqdm

from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat

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

    Parameters
    ----------
        cdl_packages : list[str]
            List of CDL packages to use.
        cdl_configs : dict[str, dict]
            Dictionary of CDL configurations.
        n_runs : int, optional
            Number of runs to generate, by default 1.
        seed : int | None, optional
            Master seed for the experiment, by default None.

    Returns
    -------
        list[dict[str, any]]
            List of configurations for the experiment.

    """
    # Generate a list of seeds for reproducibility
    rng = np.random.default_rng(seed)
    list_seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    run_config_list = []
    for package in cdl_packages:
        run_config_list.extend(
            {
                "cdl_package": package,
                "cdl_params": cdl_configs[package].copy(),
                "seed": s,
                "i": i,
            }
            for i, s in enumerate(list_seeds)
        )
    return run_config_list


@mem.cache
def run_one(
    cdl_package: str,
    cdl_params: dict,
    seed: int,
    i: int,
    data_path: Path,
    exp_dir: Path,
    reg: float,
) -> list[dict]:
    """Run a single experiment configuration."""
    logger.info("Running %s with seed %d, run %d", cdl_package, seed, i)

    data_params = np.load(data_path)
    data = data_params["X"]
    true_dict = data_params["D"]

    data = data[None, None, :, :]
    true_dict = np.expand_dims(true_dict, axis=1)

    test_data = data[:, :, 100:1000, 100:1000]
    data = data[:, :, :900, :900]

    if cdl_package == "sporco":
        other_params = cdl_params.pop("other_params")
        other_params["lmbd"] = reg
    else:
        other_params = cdl_params.copy()

    rng = np.random.default_rng(seed)
    init_dict = rng.standard_normal(
        size=(
            other_params["n_components"],
            other_params["n_channels"],
            *other_params["kernel_size"],
        )
    ).astype(np.float32)

    if cdl_package in ["rosecdl", "deepcdl"]:
        init_dict = torch.tensor(init_dict, device=other_params["device"])
        data = torch.tensor(data, device=other_params["device"])

    evaluation_model = RoseCDL(**other_params, D_init=torch.tensor(init_dict))
    eval_data = torch.tensor(data, device=other_params["device"], dtype=torch.float32)
    test_data = torch.tensor(
        test_data, device=other_params["device"], dtype=torch.float32
    )

    results = []

    xh, zh = evaluation_model.csc(eval_data)
    loss_true = evaluation_model.loss_fn(xh, zh, eval_data)
    loss_true = loss_true.item()

    txh, tzh = evaluation_model.csc(test_data)
    test_loss_true = evaluation_model.loss_fn(txh, tzh, test_data)
    test_loss_true = test_loss_true.item()

    results.append(
        {
            "name": cdl_package,
            "seed": seed,
            "epoch": 0,
            "loss": None,
            "loss_true": loss_true,
            "test_loss_true": test_loss_true,
            "time": 2e-1,
            "lmbd": reg,
        }
    )

    t_start = time.perf_counter()

    def callback_fn(model, *args):
        nonlocal t_start, evaluation_model, eval_data, test_data

        runtime = time.perf_counter() - t_start

        if cdl_package in ["rosecdl", "deepcdl"]:
            epoch, loss = args
            model_dict = torch.tensor(model.D_hat_, device=model.device)

        else:  # Sporco
            model_dict = model.getdict()[:, :, 0, :, :].transpose(3, 2, 1, 0)
            model_dict = torch.tensor(
                model_dict, device=other_params["device"], dtype=torch.float32
            )
            epoch, loss = len(results) - 1, None

        numpy_model_dict = model_dict.cpu().numpy()
        numpy_model_dict = numpy_model_dict.astype(np.float32)
        recovery_score = evaluate_D_hat(true_dict, numpy_model_dict)

        xh, zh = evaluation_model.csc(eval_data, D=model_dict)

        plt.figure()
        plt.imshow(xh[0, 0].cpu().numpy(), cmap="gray")
        plt.title(f"xh at epoch {epoch}")
        plt.savefig(exp_dir / f"xh_epoch_{epoch}_run_{i}.png")
        plt.close()

        loss_true = evaluation_model.loss_fn(xh, zh, eval_data)
        loss_true = loss_true.item()

        txh, tzh = evaluation_model.csc(test_data, D=model_dict)
        test_loss_true = evaluation_model.loss_fn(txh, tzh, test_data)
        test_loss_true = test_loss_true.item()

        results.append(
            {
                "name": cdl_package,
                "seed": seed,
                "epoch": epoch + 1,
                "recovery_score": recovery_score,
                "loss": loss,
                "loss_true": loss_true,
                "test_loss_true": test_loss_true,
                "time": runtime,
                "lmbd": reg,
            }
        )

        t_start = time.perf_counter()

    if cdl_package in ["rosecdl", "deepcdl"]:
        cdl = RoseCDL(
            **cdl_params,
            D_init=torch.tensor(init_dict),
            callbacks=[callback_fn],
        )
        if cdl_package == "deepcdl":
            cdl.csc.conv_algo = "classical"
            cdl.csc.set_conv_methods()
        cdl.fit(data)
    elif cdl_package == "sporco":
        opt_cbpdn = cbpdndl.ConvBPDNOptionsDefaults()

        opt_cbpdn["NonNegCoef"] = True
        opt_cbpdn["Verbose"] = False
        opt_cbpdn["AuxVarObj"] = False

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

    return results


if __name__ == "__main__":
    import argparse

    prs = argparse.ArgumentParser("RoseCDL 2D runtime experiment")
    prs.add_argument("data_path", type=Path, help="Path to the data file")
    prs.add_argument(
        "--n-jobs", "-j", type=int, default=1, help="Number of jobs to run in parallel"
    )
    prs.add_argument(
        "--n-runs", "-n", type=int, default=1, help="Number of runs to perform"
    )
    prs.add_argument(
        "--seed", "-s", type=int, default=None, help="Seed for the experiment"
    )
    prs.add_argument(
        "--output",
        "-o",
        default="results",
        type=Path,
        help="Output file for the results",
    )
    prs.add_argument(
        "--solver",
        type=str,
        nargs="+",  # Allow multiple solver names
        help="Filter by specific solver names",
        default=None,
    )
    prs.add_argument(
        "--gpu",
        type=str,
        default="cuda",
        help="GPU to use for the experiment (default: cuda)",
    )
    prs.add_argument("--reg", type=float, default=0.8, help="Regularization parameter")
    prs.add_argument("--debug", action="store_true", help="Debug mode")
    args = prs.parse_args()

    DEVICE = args.gpu if cuda.is_available() else "cpu"
    logger.info("Using device: %s", DEVICE)

    seed = args.seed
    if seed is None:
        rng = np.random.default_rng()
        seed = rng.integers(0, 2**32 - 1)
    logger.info("Using seed: %d", seed)

    if args.debug:
        logger.info("--- Running in debug mode ---")
        n_runs = 1
        n_jobs = 1
    else:
        n_runs = args.n_runs
        n_jobs = args.n_jobs
    logger.info("Running %d runs in parallel", n_runs)
    logger.info("Using %d jobs", n_jobs)
    reg = args.reg

    exp_dir = Path(args.output) / f"the_rosecdl_2d_runtime_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment directory: %s", exp_dir)

    cdl_packages = ["rosecdl", "deepcdl", "sporco"]
    if args.solver is not None:
        cdl_packages = args.solver
    cdl_configs = {
        "rosecdl": {
            "kernel_size": (30, 30),
            "n_channels": 1,
            "n_components": 6,
            "lmbd": reg,
            "scale_lmbd": False,
            "epochs": 10 if args.debug else 50,
            "max_batch": 20,
            "mini_batch_size": 10,
            "sample_window": 960,
            "optimizer": "adam",
            "n_iterations": 40 if args.debug else 60,
            "window": False,
            "device": DEVICE,
            "positive_D": True,
            "outliers_kwargs": None,
        },
        "sporco": {
            "other_params": {
                "kernel_size": (30, 30),
                "n_channels": 1,
                "n_components": 6,
                "lmbd": reg,
                "scale_lmbd": False,
                "epochs": 10 if args.debug else 50,
                "n_iterations": 40 if args.debug else 60,
                "window": False,
                "device": DEVICE,
                "positive_D": True,
            },
            "lmbda": 0.8,
            "n_iter": 5 if args.debug else 300,
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
    logger.info("Generated %d run configurations", len(run_configs))

    import json

    config_to_save = {
        "cdl_configs": cdl_configs,
        "n_runs": n_runs,
        "seed": seed,
        "args": vars(args),
    }
    with (exp_dir / "experiment_config.json").open("w") as f:
        json.dump(config_to_save, f, indent=4, default=str)

    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(
            **run_config,
            data_path=args.data_path,
            exp_dir=exp_dir,
            reg=args.reg,
        )
        for run_config in run_configs
    )
    results = [r for res in tqdm(results, "Run", total=len(run_configs)) for r in res]

    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / f"df_results_{cdl_packages}.csv", index=False)
    logger.info("Results saved to %s", exp_dir / f"df_results_{cdl_packages}.csv")
    logger.info("Experiment finished")

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

    # Test loss for the different methods
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for name in df_results.name.unique():
        # Group by name and epoch, then calculate median and quantiles
        name_data = df_results[df_results.name == name].groupby("epoch")

        # Calculate the median curve
        median_curve = name_data[["time", "test_loss_true"]].median()
        median_curve["time"] = median_curve["time"].cumsum()
        median_curve["test_loss_true"] = (
            median_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        # Calculate the 0.2 and 0.8 quantiles
        q02_curve = name_data[["time", "test_loss_true"]].quantile(0.2)
        q02_curve["time"] = q02_curve["time"].cumsum()
        q02_curve["test_loss_true"] = (
            q02_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        q8_curve = name_data[["time", "test_loss_true"]].quantile(0.8)
        q8_curve["time"] = q8_curve["time"].cumsum()
        q8_curve["test_loss_true"] = (
            q8_curve["test_loss_true"] - df_results["test_loss_true"].min() + 1e1
        )

        (line,) = ax.plot(
            median_curve["time"], median_curve["test_loss_true"], label=name
        )
        color = line.get_color()
        ax.fill_between(
            median_curve["time"],
            q02_curve["test_loss_true"],
            q8_curve["test_loss_true"],
            color=color,
            alpha=0.2,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Loss")
    plt.title("Test performance of the different methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "test_loss_true.png")
    plt.show()
