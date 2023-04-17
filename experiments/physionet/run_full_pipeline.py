""" For a set of hyper parameters, run the CDL on the pre-determined population,
plot its atoms, compute the recovery DataFrame and plot the final figure.
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--group",
    type=str,
    choices=['a', 'b', 'c', 'x'],
    help="group id to run the CDL on ('a': apnea, 'b': borderline apnea, 'c': control, 'x': test)",
    default='a',
)
parser.add_argument("--n_atoms", type=int, default=3)
parser.add_argument("--n_times_atom", type=int, default=75)
parser.add_argument("--n_iter", type=int, default=200)
parser.add_argument("--fit", type=str, default='N', choices=['N', 'A'],
                    help="'A': apnea minutes, 'N': non-apnea minutes")
parser.add_argument("--add_number", action="store_true", help="add number of trials")
parser.add_argument("--type", type=str, default='box', choices=['box', 'violin'])

args = parser.parse_args()
group_id = args.group

for group_id in ['a', 'b', 'c', 'x']:
    print("Group id:", group_id)
    # population
    os.system("python run_pop_cdl.py "
            "--group {}"
            .format(group_id)
    )
    os.system("python get_pop_recovery.py "
            "--group {}"
            .format(group_id)
    )
    for fit in ['N', 'A']:
        if fit == 'A' and group_id in ['C', 'X']:
            continue
        print("Fit on:", fit)
        # individual
        os.system("python run_apnea.py "
                "--group {} --n_atoms {} --n_times_atom {} --n_iter {} --fit {}"
                .format(group_id, args.n_atoms, args.n_times_atom, args.n_iter, fit)
        )
        os.system("python plot_group_atoms.py "
                "--group {} --fit {}"
                .format(group_id, fit)
        )
        os.system("python get_recovery.py "
                "--group {} --fit {}"
                .format(group_id, fit)
        )
        # final plot
        os.system("python plot_recovery.py "
                "--group {} --fit {}  --type {} --add_number"
                .format(group_id, fit, args.type)
        )