"""This file contains the code to run WinCDL on 2D image data."""

from pathlib import Path
import numpy as np
from torch import cuda

from wincdl.utils.utils_exp import evaluate_D_hat, make_size
from wincdl.wincdl import WinCDL

import time


EXP_DIR = Path("results")
EXP_DIR.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run WinCDL on 2D image data")
    parser.add_argument(
        "data_path", type=str, help="Path to NPZ file containing the data"
    )
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--reg", type=float, default=0.8, help="Regularization parameter"
    )
    args = parser.parse_args()

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    seed = args.seed if args.seed is not None else np.random.randint(0, 2**32 - 1)

    exp_dir = EXP_DIR / f"wincdl_2d_reg_{args.reg}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # WinCDL parameters for 2D processing
    wincdl_params = {
        "n_components": 6,
        "kernel_size": (35, 35),
        "n_channels": 1,
        "lmbd": args.reg,
        "scale_lmbd": True,
        "epochs": 50,
        "max_batch": 20,
        "mini_batch_size": 10,
        "sample_window": 500,
        "optimizer": "linesearch",
        "n_iterations": 50,
        "window": True,
        "device": DEVICE,
    }

    # Load the data
    data = np.load(args.data_path)
    X = data.get("X")
    D_true = data.get("D")

    if X.ndim == 2:
        X = X[np.newaxis, np.newaxis, :, :]

    # Setup the callback for monitoring and visualization
    results = []
    t_start = time.perf_counter()

    def callback_fn(model, epoch, loss):
        global t_start
        runtime = time.perf_counter() - t_start
        D_true_resized = make_size(D_true, model.D_hat_.shape)
        score = evaluate_D_hat(D_true_resized, model.D_hat_)

        results.append(
            {
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "score": score,
                "time": runtime,
            }
        )
        t_start = time.perf_counter()

    # Run WinCDL
    wincdl = WinCDL(**wincdl_params, callbacks=[callback_fn])
    wincdl.fit(X)

    print(results)
