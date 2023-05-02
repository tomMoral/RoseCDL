# %%
import numpy as np
from pathlib import Path

from alphacsc.utils.dictionary import get_uv
from dripp.cdl.plotting import display_atoms

from experiments.scripts.utils import \
    get_lambda_global, get_D_sub, get_subject_z_and_cost, get_camcan_info

N_ATOMS = 70
N_TIMES_ATOM = 150


DATA_PATH = Path("/storage/store2/work/bmalezie/camcan-cdl")
cat = 1
data_path_cat = DATA_PATH/f'cat{cat}'
all_paths = [x for x in data_path_cat.glob('**/*') if x.is_file()]

D_hat = np.load('../scripts/D_hat_cat1.npy')
N_CHANNELS = 204

subject_id = all_paths[0].name.split('.')[0]
info = get_camcan_info(subject_id)

# uv = get_uv(D_hat)
dict_cdl = dict(u_hat_=D_hat[:, :N_CHANNELS],
                v_hat_=D_hat[:, N_CHANNELS:])
plotted_atoms = list(range(N_ATOMS))
display_atoms(dict_cdl, info, plotted_atoms, sfreq=150.,
              plot_spatial=True, plot_temporal=True, plot_psd=False,
              save_plot=True,
              fig_name=f"atoms_cat{cat}.pdf",
              dir_path='', dpi=300)
# %%
