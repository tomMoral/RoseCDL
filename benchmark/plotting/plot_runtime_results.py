import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# Load the data and resample the curves on a regular grid to allow computing
# the runtime.
df = pd.read_parquet("outputs/benchmark_runtime.parquet")
Ts = [int(T) for T in df['p_dataset_n_times'].dropna().unique()]
batch_params = df.groupby('p_solver_sample_window').first()['p_solver_mini_batch_size']
batch_params = [(int(w), int(bs)) for w, bs in batch_params.items()]
regs = df['p_obj_reg'].dropna().unique()
regs.sort()

col = 'objective_loss_val'
groups = [
    'solver_name', 'p_obj_reg', 'p_dataset_n_times', 'p_dataset_random_state'
]

# Interpolate the curves to a regular grid and rescale between 1e-3 and 1
t = np.logspace(np.log10(df['time'].min()), np.log10(df['time'].max()), 1000)
curves = df.groupby(groups).apply(
    lambda x: pd.DataFrame({
        't': np.clip(t, 1, None),
        'loss_val': (interp1d(
            x['time'], x[col], bounds_error=False,
            fill_value=tuple(x[col].iloc[[0, -1]])
        )(t) - x[col].min()) / (x[col].iloc[0] - x[col].min()) + 1e-3,
    })
).reset_index(level=-1, drop=True)

# Compute the runtimes as the first time the loss reaches 1% of the final value
# and compute median/quantiles over the seeds.
runtimes = curves[
    (curves - curves.groupby(groups).last())['loss_val'] < 1e-2
].reset_index().groupby(groups).first()['t']
runtimes = runtimes.groupby(groups[:-1]).median()
# compute median and quantiles for the curves
med_curves = (
    curves.reset_index().groupby(groups[:-1] + ['t'])['loss_val']
    .quantile([0.5, 0.2, 0.8]).unstack()
)
# Compute the final loss as the median of the last value of the curves
loss = df.groupby(groups).last()[col].groupby(groups[:-1]).median()
table = pd.concat([loss, runtimes], axis=1)

# Display the results with the computed runtimes and the median curves
# to validate the results.
reg = 0.8
solvers = {}
solvers['linesearch'] = [('alphaCSC[outliers_kwargs=None,type=batch]', 'alphacsc')]
# solvers += [('alphaCSC[outliers_kwargs=None,type=online]', 'alphacsc-o')]
template_rosecdl = (
    'RoseCDL[mini_batch_size={bs},n_csc_iterations=50,optimizer={opt},'
    'outliers_kwargs=None,random_state=None,sample_window={win}]'
)
solvers['linesearch'] += [
    (template_rosecdl.format(opt='linesearch', win=win, bs=bs),
     f'rosecdl[{win}]') for win, bs in batch_params
]
solvers['adam'] = [
    (s.replace('linesearch', 'adam'), l) for s, l in solvers['linesearch']
]
n_solvers = len(solvers['adam'])

fig, axes = plt.subplots(2, len(Ts), figsize=(8, 7))
for k, (label, l_solvers) in enumerate(solvers.items()):
  for i, t in enumerate(Ts):
    for j, (solver, l) in enumerate(l_solvers):
        med_curves.loc[(solver, reg, t)].reset_index().plot(
            x='t', y=0.5, logx=True, logy=True, ax=axes[k, i], label=l
        )
        axes[k, i].vlines(runtimes.loc[(solver, reg, t)], 0, 1, f'C{j}')
        if i == len(Ts) // 2:
            axes[k, i].set_title(label)
fig.savefig("med_curves.pdf")

# Now plot the runtime results
cmap = plt.get_cmap('viridis', n_solvers)

fig, axes = plt.subplots(2, 1, height_ratios=[0.05, 1], figsize=(6, 3.5))
ax = axes[1]

wd = (1 - 0.3 ) / n_solvers
for l, T in enumerate(reversed(Ts)):
  ax = axes[1]
  for k, l_solvers in enumerate(solvers.values()):
    for i, reg in enumerate(regs):
      for j, (solver, label) in enumerate(l_solvers):
        ax.add_artist(plt.Rectangle(
            (k * len(regs) + i + j * wd, 0),
            wd*0.9, runtimes.loc[(solver, reg, T)],
            color=cmap(j), alpha=0.5
        ))
  ax.set_yscale('log')
  ax.set_xlim(0, k*len(regs) + i+1)
  ax.set_ylim(1, runtimes.max() *1.2)

ticks = [rf"$\lambda={reg}$" for reg in regs]
ticks = [
    t if '0.3' not in t else f'{t}\n{opt}'
    for opt in solvers for t in ticks
]
plt.xticks(
    [i + 0.7 / 2 for i in range(len(regs) * 2)],
    ticks
)
handles, labels = list(zip(*[
    (plt.Rectangle((0, 0), 1, 1, color=cmap(i)), label)
    for i, (_, label) in enumerate(solvers['adam'])
]))
legend = axes[0].legend(
    handles=handles, labels=labels,  ncols=n_solvers, loc='upper center'
)
handles, labels = list(zip(*[
    (plt.Rectangle((0, 0), 1, 1, color='k', alpha=0.5 +0.25 * k), f"T={T // 1000}k")
    for T in Ts
]))
axes[0].legend(
    handles=handles, labels=["T=10k", "T=30k", "T=100k"], loc="lower center",
    ncols=len(Ts)
)
axes[0].add_artist(legend)
axes[0].set_axis_off()
plt.yscale('log')

plt.xlim(0, k*len(regs) + i+1)
plt.ylim(10, runtimes.max() *1.2)
fig.savefig("runtimes.pdf")

reg, T = 0.8, 100_000

fig, ax = plt.subplots()
markers = {"adam": "s", "linesearch": "^"}
for opt, l_solvers in solvers.items():
  m = markers[opt]
  for i, (solver, label) in enumerate(l_solvers):
    ax.scatter(runtimes.loc[(solver, reg, T)], loss.loc[(solver, reg, T)], color=cmap(i), label=label, marker=m)

handles, labels = list(zip(*[
    (plt.Rectangle((0, 0), 1, 1, color=cmap(i)), l)
    for i, (_, l) in enumerate(solvers['adam'])]))
legend = ax.legend(handles, labels)
handles, labels = list(zip(*[
    (plt.Line2D([], [], marker=m, linestyle='None', color="k"), l)
    for l, m in markers.items()
]))
ax.legend(handles, labels, loc="right")
ax.add_artist(legend)

ax.set_xlabel("Runtime [s]")
ax.set_ylabel("Final Loss")
ax.set_yscale('log')
fig.tight_layout()

fig.savefig("runtime_vs_loss.pdf")

# display summary table for the rebuttal
table_ratio = (
    table / table.loc[solvers['adam'][0][0]]
).query("solver_name.str.contains('RoseCDL') and p_dataset_n_times > 30000 and p_obj_reg == 0.8").reset_index()
table_ratio['opt'] = table_ratio['solver_name'].apply(lambda c: c.split("optimizer=")[-1].split(',')[0]).values
table_ratio['win'] = table_ratio['solver_name'].apply(lambda c: int(c.split("sample_window=")[-1].split(']')[0])).values
table_ratio = table_ratio.sort_values(by=['opt', 'win'])
table_final = table_ratio.pivot(columns=['opt', 'win'], index=['p_dataset_n_times', 'p_obj_reg']).drop(columns=['solver_name'])
table_final.loc[:, 'objective_loss_val'] = (table_final.loc[:, 'objective_loss_val'].values - 1)

print((table_final.median().unstack(level=[1, 2])).apply(lambda x: [f"{v:.1%}" for v in x]))

plt.show()
