"""This file contains the code to run RoseCDL on 2D image data."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import cuda

from rosecdl.rosecdl import RoseCDL
from rosecdl.utils.utils_exp import evaluate_D_hat

EXP_DIR = Path("results")
EXP_DIR.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RoseCDL on 2D image data")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to NPZ file containing the data. This data can be generated "
        "using the `experiments/letters/generate_letters.py` script.",
    )
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--reg", type=float, default=0.8, help="Regularization parameter"
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else np.random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")

    exp_dir = EXP_DIR / f"rosecdl_2d_reg_{args.reg}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Load the data
    data = np.load(args.data_path)
    X = data.get("X")
    D_true = data.get("D")

    if X.ndim == 2:
        X = X[None, None]

    # Setup the callback for monitoring and visualization
    results = []
    t_start = time.perf_counter()

    def callback_fn(model, epoch, loss):
        global t_start
        runtime = time.perf_counter() - t_start
        score = evaluate_D_hat(D_true[:, None], model.D_hat_)

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

    # Run RoseCDL
    rosecdl = RoseCDL(
        n_components=6,
        kernel_size=(35, 35),
        n_channels=1,
        lmbd=args.reg,
        scale_lmbd=True,
        epochs=30,
        max_batch=1,
        mini_batch_size=1,
        sample_window=500,
        optimizer="linesearch",
        n_iterations=50,
        window=True,
        device="cuda" if cuda.is_available() else "cpu",
        callbacks=[callback_fn],
        random_state=seed,
    )
    rosecdl.fit(X)

    results = pd.DataFrame(results)
    results["time"] = results["time"].cumsum()
    print(results)

    # Plot learning curve and recovery score
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    results.plot(x="time", y="loss", ax=ax[0], title="Loss")
    results.plot(x="time", y="score", ax=ax[1], title="Recovery score")
    fig.savefig(exp_dir / "learning_curve.png")

    # Plot atoms learned
    fig, ax = plt.subplots(1, rosecdl.D_hat_.shape[0], figsize=(10, 5))
    for i, atom in enumerate(rosecdl.D_hat_):
        ax[i].imshow(atom.squeeze(), cmap="gray")
        ax[i].axis("off")

    plt.savefig(exp_dir / "atoms_learned_rosecdl.png")
