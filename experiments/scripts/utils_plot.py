import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def unsplit_z(z):
    """

    """
    n_splits, n_channels, n_times_split = z.shape
    z = z.swapaxes(0, 1).reshape(
        1, n_channels, n_times_split * n_splits)

    return z


def apply_threshold(z, p_threshold, per_atom=True):
    if len(z.shape) == 3:
        if z.shape[0] > 1:
            z = unsplit_z(z)
        z = z[0]

    n_atoms = z.shape[0]

    if per_atom:
        z_nan = z.copy()
        z_nan[z_nan == 0] = np.nan
        mask = z_nan >= np.nanpercentile(
            z_nan, p_threshold, axis=1, keepdims=True)

        return [z_nan[i][mask[i]] for i in range(n_atoms)]

    else:
        threshold = np.percentile(z[z > 0], p_threshold)  # global threshold
        print(f"Global thresholding at {p_threshold}%: {threshold}")
        return [this_z[this_z > threshold] for this_z in z]


def plot_z_boxplot(z_hat, p_threshold=0, per_atom=True,
                   yscale='log', add_points=True, add_number=True,
                   fig_name=None):
    """
    Plot activations boxplot for each atom, with a possible thresholding.

    Parameters
    ----------

    Returns
    -------
    """
    n_atoms = z_hat.shape[1]
    values = apply_threshold(
        z=z_hat, p_threshold=p_threshold, per_atom=per_atom)

    df_z = pd.DataFrame(data=values).T
    df_z = df_z.rename(columns={k: f'Values{k}' for k in range(n_atoms)})
    df_z["id"] = df_z.index
    df_z = pd.wide_to_long(df_z, stubnames=['Values'], i='id', j='Atom')\
             .reset_index()[['Atom', 'Values']]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale(yscale)

    sns.boxplot(x='Atom', y='Values', data=df_z)

    if add_points:
        sns.stripplot(
            x="Atom", y="Values", data=df_z, size=2, color=".3", linewidth=0)

    if add_number:
        ax2 = ax.twinx()
        xx = list(range(n_atoms))
        yy = [len(z) for z in values]
        ax2.plot(xx, yy, color="black", alpha=.8)
        ax2.set_ylim(0)
        ax2.set_ylabel('# non-nul activations')

    plt.xticks(rotation=45)
    title = "Activations repartition"
    if p_threshold > 0:
        title += f" with {'per-atom' if per_atom else 'global'} thresholding of {p_threshold}%"
    plt.title(title)

    if fig_name:
        plt.savefig(fig_name)

    plt.show()
