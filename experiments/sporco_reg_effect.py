"""This script analyzes the effect of regularization parameter on dictionary learning
using SPORCO for letter data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib import Memory
import sporco.dictlrn.cbpdndl as cbpdndl
from rosecdl.utils.utils_exp import evaluate_D_hat

# Setup cache and directories
mem = Memory(location="__cache__", verbose=0)
EXP_DIR = Path("results")
EXP_DIR.mkdir(exist_ok=True, parents=True)


@mem.cache
def run_one(reg, run_idx, opt, seed, data_path):
    """Run a single experiment with SPORCO."""
    # Set random seed
    np.random.seed(seed + run_idx)

    # Load data
    data = np.load(data_path)
    X = data.get("X")
    D_true = data.get("D")

    # Reshape data for SPORCO format
    X = X[None, None, ...]
    D_true = np.expand_dims(D_true, axis=1)

    # Crop data if needed
    X = X[:, :, :800, :800]

    # Prepare data for SPORCO (transpose to match SPORCO's format)
    X_sporco = X.transpose(3, 2, 1, 0).copy()

    # Initialize dictionary for SPORCO format
    D_init = np.random.randn(
        opt['kernel_size'][1],
        opt['kernel_size'][0],
        opt['n_channels'],
        opt['n_components']
    )

    results = []

    def callback(obj):
        """Store iteration stats."""
        # Get dictionary in correct format for evaluation
        D_hat = obj.getdict()[:, :, 0, :, :].transpose(3, 2, 1, 0)
        recovery_score = evaluate_D_hat(D_true, D_hat)
        results.append({
            'run_idx': run_idx,
            'epoch': obj.getitstat().Iter[-1],
            'loss': obj.getitstat().ObjFun[-1],
            'recovery_score': recovery_score,
            'reg': reg,
        })

    # Setup SPORCO options
    opt_cbpdn = cbpdndl.ConvBPDNOptionsDefaults()
    opt_cbpdn['NonNegCoef'] = True
    opt_cbpdn['Verbose'] = False
    opt_cbpdn['AuxVarObj'] = False

    solver_opt = cbpdndl.ConvBPDNDictLearn.Options({
        'Verbose': False,
        'MaxMainIter': opt['n_iter'],
        'CBPDN': opt_cbpdn,
        'Callback': callback
    }, dmethod='cns')

    # Create and run solver
    xp = cbpdndl.ConvBPDNDictLearn(
        D0=D_init,
        S=X_sporco,
        lmbda=reg,
        opt=solver_opt,
        dmethod='cns',
        dimN=2  # 2D data
    )
    xp.solve()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze regularization effect using SPORCO"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the data file (NPZ format with keys 'X' and 'D')",
    )
    parser.add_argument(
        "--n-jobs", "-j", type=int, default=1,
        help="Number of parallel jobs"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Master seed for reproducible job"
    )
    parser.add_argument(
        "--n-runs", "-n", type=int, default=5,
        help="Number of repetitions for the experiment"
    )
    args = parser.parse_args()

    # Set random seed
    seed = args.seed if args.seed is not None else np.random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")

    # SPORCO parameters
    opt = {
        'n_iter': 125,
        'kernel_size': (35, 35),
        'n_channels': 1,
        'n_components': 6,
        'NonNegCoef': True,
        'ZeroMean': False,
    }

    # Experiment parameters
    reg_values = np.linspace(1, 2.2, 20)
    exp_dir = EXP_DIR / "sporco_reg_effect"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Run experiments
    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one)(reg, run_idx, opt, seed, data_path=args.data_path)
        for reg in reg_values
        for run_idx in range(args.n_runs)
    )

    # Process and save results
    results = [r for res in tqdm(results, "Processing results") for r in res]
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_dir / "sporco_reg_effect.csv", index=False)

    print("\nExperiment completed! Results saved to sporco_reg_effect.csv")
