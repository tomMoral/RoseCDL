
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

outliers, reg = True, 0.8

df = pd.read_parquet("benchmark/outputs/benchmark_outliers.parquet")
df["cls_name"] = df.apply(lambda s: s["solver_name"].split("[")[0] + ('' if 'outliers_kwargs={' not in s['solver_name'] else 'OD'), axis=1)

order = ['Sporco', 'SporcoOD', 'alphaCSC', 'alphaCSCOD', 'RoseCDL', 'RoseCDLOD']
fig = plt.figure(figsize=(6.75, 2.5))
gs = plt.GridSpec(2, 2, height_ratios=[0.1, 0.9])
ax = fig.add_subplot(gs[1, 0])
ax_plot = fig.add_subplot(gs[1, 1])
Ts = sorted([int(T) for T in df['p_dataset_n_times'].unique()])
for i, T in enumerate(Ts):
    print(i, T)
    this_df = df
    this_df = df.query("solver_name.str.contains('outliers_kwargs=None') or solver_name.str.contains('mad')")

    t = np.linspace(0, 150, 400)
    tt = (
        this_df.query("p_dataset_n_times == @T and p_dataset_contamination == @outliers")
        .groupby(["solver_name", "p_dataset_random_state", "p_obj_reg", "cls_name"])[['time', 'objective_recovery_score']]
        .apply(lambda d: pd.DataFrame(dict(time=t, val=interp1d(d['time'], d['objective_recovery_score'], fill_value=tuple(d['objective_recovery_score'].iloc[[0, -1]]), bounds_error=False)(t))))
    ).reset_index()

    hist = tt.groupby(["solver_name", "p_obj_reg", "cls_name", "time"]).median().reset_index().groupby(['solver_name', 'p_obj_reg']).last().reset_index().groupby(["cls_name", "p_obj_reg"]).apply(lambda g: g.loc[g['val'].idxmax()], include_groups=False).reset_index()
    vals = hist.query("p_obj_reg == @reg")
    if order is None:
       vals = vals.sort_values('val')
       order = vals['cls_name']
    vals = vals.set_index('cls_name').loc[order, 'val']

    for j, (name, v) in enumerate(zip(vals.index, vals)):
        print(i, j, name)
        ax.add_artist(plt.Rectangle((i +  0.1 * j, 0), 0.09, v, color=f"C{j}", hatch='\\' if 'OD' in name else None))

    if i == len(Ts) - 1:
        curves = (
            tt[tt['solver_name'].isin(hist['solver_name'])]
            .query("p_obj_reg == @reg")
            .groupby(["cls_name", "time"])[['val']]
            .quantile([0.2, 0.5, 0.8]).unstack().droplevel(0, axis=1)
        )
        for j, name in enumerate(order):
            c = curves.loc[name]
            ax_plot.fill_between(c.index, c[0.2], c[0.8], alpha=0.3, color=f"C{j}")
            ax_plot.semilogx(c.index, c[0.5], color=f"C{j}", lw=2)

ax.set_xlim(0, len(Ts) - 1 + len(order) * .1)
ax.set_xticks([i + 0.25 for i in range(len(Ts))], [f"{T=}" for T in Ts])
ax.set_ylabel("Recovery score")

ax_plot.set_xlabel("Time [sec]")

ax_legend = fig.add_subplot(gs[0, :])
ax_legend.set_axis_off()
handles = [plt.Rectangle((0, 0), 0, 0, color=f"C{j}") for j in range(len(order))]
ax_legend.legend(handles, order, ncols=3, loc='center')

plt.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.18, hspace=.3)
plt.savefig("results/outlier_recovery_1D.pdf")
