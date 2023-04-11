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

os.system("python run_apnea.py "
          "--group {} --n_atoms {} --n_times_atom {} --n_iter {} --fit {}"
          .format(args.group, args.n_atoms, args.n_times_atom, args.n_iter, args.fit)
)
os.system("python plot_group_atoms.py "
          "--group {} --fit {}"
          .format(args.group, args.fit)
)
os.system("python get_recovery.py "
          "--group {} --fit {}"
          .format(args.group, args.fit)
)
os.system("python plot_recovery.py "
          "--group {} --fit {}  --type {} --add_number"
          .format(args.group, args.fit, args.type)
)