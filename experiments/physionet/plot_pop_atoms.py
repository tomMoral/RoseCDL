import numpy as np
import argparse

from utils_apnea import plot_temporal_atoms

parser = argparse.ArgumentParser()
parser.add_argument(
    "--group",
    type=str,
    choices=['a', 'b', 'c', 'x'],
    help="group id to run the CDL on ('a': apnea, 'b': borderline apnea, 'c': control, 'x': test)",
)

args = parser.parse_args()
group_id = args.group

# load population dictionary
d_hat_pop = np.load(f'd_hat_pop_{group_id}.npy').squeeze()

plot_temporal_atoms(
    d_hat_pop, save_fig=f'pop_{group_id}_atoms.pdf')