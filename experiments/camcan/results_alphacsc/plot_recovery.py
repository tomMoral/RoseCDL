# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

lmbd_max = "fixed"
for reg in [0.1, 0.3]:
    suff = f"reg_{reg}_lmbd_{lmbd_max}"

    df_cost = pd.read_csv(f"df_cost_{suff}.csv")
    df_cost["subject_id"] = df_cost["subject_id"].apply(lambda x: x[4:])

    # get sub-dataframe for D_init
    d_init_index = df_cost[df_cost["dict_fit"] == "D_init"].index
    df_init = df_cost.loc[d_init_index.values]

    df_cost.drop(d_init_index.values, inplace=True)

    # get sub-dataframe for D_random
    d_random_index = df_cost[df_cost["dict_fit"] == "D_random"].index
    df_random = df_cost.loc[d_random_index.values]

    df_cost.drop(d_random_index.values, inplace=True)
    df_cost["dict_fit"] = df_cost["dict_fit"].apply(lambda x: x[4:])

    # get sub-dataframe for D_self
    self_index = df_cost[df_cost["subject_id"] == df_cost["dict_fit"]].index
    df_self = df_cost.loc[self_index.values]
    df_cost.drop(self_index.values, inplace=True)

    df_boxplot = df_cost[df_cost["subject_id"] != df_cost["dict_fit"]]
    g = sns.boxplot(data=df_cost, x="subject_id", y="cost")

    xticklabels = [t.get_text() for t in g.get_xticklabels()]

    yy_self = [
        df_self[df_self["subject_id"] == xlabel]["cost"].values[0]
        for xlabel in xticklabels
    ]
    plt.scatter(xticklabels, yy_self, marker="*", label="self")
    # add chunck init
    yy_init = [
        df_init[df_init["subject_id"] == xlabel]["cost"].values[0]
        for xlabel in xticklabels
    ]
    plt.scatter(xticklabels, yy_init, marker="v", label="init")
    # add random init
    yy_random = [
        df_random[df_random["subject_id"] == xlabel]["cost"].values[0]
        for xlabel in xticklabels
    ]
    plt.scatter(xticklabels, yy_random, marker="v", label="random")

    plt.xticks(rotation=90)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(suff.replace("_", " "))
    plt.xlabel("Subject ID")
    plt.ylabel("Lasso cost")
    plt.tight_layout()
    plt.savefig(f"alphacsc_recovery_{suff}.pdf", dpi=300)
    plt.show()
    plt.clf()

# %%
