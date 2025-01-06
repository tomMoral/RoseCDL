"""
This file generates data, process it into random windows,
then applies different basic anomaly detection methods
(Percentile, IQR, Z-Score, MAD) to discard anomalous windows
and only learn from normal windows. Finally, the recovery
score for the dictionary learned is computed.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from simulate import simulate_data2
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    SignalDataset,
    filter_iqr,
    filter_mad,
    filter_percentile,
    filter_zscore,
)

from wincdl.wincdl import WinCDL

# ===== Load config =====
config_path = Path(__file__).parent / "config_recovery_ad.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# ===== Initialize results dictionary =====
results = {
    "percentile": [],
    "iqr": [],
    "zscore": [],
    "mad": [],
    "original": [],
    "true": [],
}

# ===== Create output directory =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent.parent / "output" / "results"
output_dir.mkdir(exist_ok=True)

# ===== Main loop =====
total_iterations = cfg["training"]["n_runs"] * 5  # 5 types of data
pbar = tqdm(total=total_iterations, desc="Processing")

# Loop over runs
for run in range(cfg["training"]["n_runs"]):
    print(f"Run {run + 1}/{cfg['training']['n_runs']}")

    # ===== Generate new synthetic data for each run =====
    X, ds, _ = simulate_data2(
        n_trials=cfg["data"]["n_trials"],
        n_times=cfg["data"]["n_times"],
        n_times_atom=cfg["data"]["n_times_atom"],
        n_atoms=cfg["data"]["n_atoms"],
        window=cfg["data"]["window"],
        overlap=cfg["data"]["overlap"],
        random_state=cfg["data"]["random_state"] + run,  # Different seed each run
        p_acti=cfg["data"]["p_acti"],
        p_contaminate=cfg["data"]["p_contaminate"],
    )
    results["true"].append(ds)

    # X shape: (n_trials, n_channels, n_times)
    X = X[:, np.newaxis, :]  # Add channel

    # Flatten X then window it
    X_flat = X.flatten()

    # ===== Sample random windows =====
    rng = np.random.RandomState(
        cfg["windows"]["random_state"] + run
    )  # Different seed each run
    window_size = cfg["windows"]["window_size"]
    stride = cfg["windows"]["stride"]
    n_windows = cfg["windows"]["n_windows"]

    n_possible_windows = ((X_flat.shape[0] - window_size) // stride) + 1
    print(X.shape, window_size, stride, n_windows, X_flat.shape)
    print(f"Total possible windows: {n_possible_windows}")
    window_starts = rng.choice(n_possible_windows, size=n_windows, replace=True)
    X_windows = np.array(
        [X_flat[start : start + window_size] for start in window_starts * stride]
    )

    X_windows = X_windows.reshape(-1, 1, window_size)

    # ===== Filter windows using anomaly detection methods =====
    normal_windows_percentile = filter_percentile(
        X_windows, X, cfg["anomaly_detection"]["percentile"]
    )
    normal_windows_iqr = filter_iqr(X_windows, X, cfg["anomaly_detection"]["iqr"])
    normal_windows_zscore = filter_zscore(
        X_windows, X, cfg["anomaly_detection"]["zscore"]
    )
    normal_windows_mad = filter_mad(X_windows, X, cfg["anomaly_detection"]["mad"])

    # ===== Create datasets and dataloaders =====
    batch_size = cfg["training"].get("batch_size", 32)
    original_dataset = SignalDataset(X_windows)
    percentile_dataset = SignalDataset(normal_windows_percentile)
    iqr_dataset = SignalDataset(normal_windows_iqr)
    zscore_dataset = SignalDataset(normal_windows_zscore)
    mad_dataset = SignalDataset(normal_windows_mad)

    print(
        len(original_dataset),
        len(percentile_dataset),
        len(iqr_dataset),
        len(zscore_dataset),
        len(mad_dataset),
    )

    original_loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=True)
    percentile_loader = DataLoader(
        percentile_dataset, batch_size=batch_size, shuffle=True
    )
    iqr_loader = DataLoader(iqr_dataset, batch_size=batch_size, shuffle=True)
    zscore_loader = DataLoader(zscore_dataset, batch_size=batch_size, shuffle=True)
    mad_loader = DataLoader(mad_dataset, batch_size=batch_size, shuffle=True)

    # ===== WinCDL on each dataset =====
    for name, loader in [
        ("original", original_loader),
        ("percentile", percentile_loader),
        ("iqr", iqr_loader),
        ("zscore", zscore_loader),
        ("mad", mad_loader),
    ]:
        pbar.set_description(f"Run {run + 1} - Data {name}")
        model = WinCDL(
            n_components=int(cfg["model"]["n_components"]),
            kernel_size=int(cfg["model"]["kernel_size"]),
            n_channels=int(cfg["model"]["n_channels"]),
            lmbd=cfg["model"]["lmbd"],
            n_iterations=cfg["model"]["n_iterations"],
            epochs=cfg["model"]["epochs"],
            window=False,
            list_D=True,
        )
        losses, list_D, times = model.fit(loader)
        results[name].append(list_D)
        pbar.update(1)

pbar.close()

# Save results
output_file = output_dir / f"d_hat_results_{timestamp}.pkl"
with open(output_file, "wb") as f:
    pickle.dump(results, f)

print(f"Results saved to {output_file}")
