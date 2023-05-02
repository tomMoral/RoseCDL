# %%
import pandas as pd
from pathlib import Path
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--group",
    type=str,
    choices=['a', 'b', 'c', 'x'],
    help="group id to run the CDL on ('a': apnea, 'b': borderline apnea, 'c': control, 'x': test)",
)
parser.add_argument("--fit", type=str, default='N')
parser.add_argument("--add_number", action="store_true", help="add number of trials")
parser.add_argument("--type", type=str, default='box', choices=['box', 'violin'])

args = parser.parse_args()
group_id = args.group
fit_on = args.fit
add_number = args.add_number
type = args.type

group_des = dict(a='apnea', b='borderline apnea', c='control', x='test')

if add_number:
    participants = pd.read_csv(
        Path("apnea-ecg/participants.tsv"), sep='\t')
    subject_id_list = participants['Record'].values
    subject_id_list = [id for id in subject_id_list if id[0] == group_id]
    n_non_apnea = participants[participants['Record'].isin(subject_id_list)]['non-apn (minutes)'].values

# load recovery DataFrame
df_cost = pd.read_csv(f'recovery_df_{group_id}_{fit_on}.csv')
# try load population recovery DataFrame
try:
    df_pop_cost = pd.read_csv(f'recovery_pop_df_{group_id}.csv')
except:
    df_pop_cost = None


# get sub-dataframe for D_init
d_init_index = df_cost[df_cost['dict_fit'] == 'D_init'].index
df_init = df_cost.loc[d_init_index.values]
df_cost.drop(d_init_index.values, inplace=True)

# get sub-dataframe for D_random
d_random_index = df_cost[df_cost['dict_fit'] == 'D_random'].index
df_random = df_cost.loc[d_random_index.values]
df_cost.drop(d_random_index.values, inplace=True)

# get sub-dataframe for D_self
self_index = df_cost[df_cost['subject_id'] == df_cost['dict_fit']].index
df_self = df_cost.loc[self_index.values]
df_cost.drop(self_index.values, inplace=True)

# plot recovery boxplot
fig, ax = plt.subplots()
ax.set_yscale('log')

df_boxplot = df_cost[df_cost['subject_id'] != df_cost['dict_fit']]
sns.set_palette("colorblind")
if type == "box":
    g = sns.boxplot(data=df_cost, x="subject_id", y="cost")
elif type == "violin":
    g = sns.violinplot(data=df_cost, x="subject_id", y="cost")

xticklabels = [t.get_text() for t in g.get_xticklabels()]

yy_self = [df_self[df_self['subject_id'] == xlabel]['cost'].values[0]
           for xlabel in xticklabels]
plt.scatter(xticklabels, yy_self, marker='*', label='self')
# add chunck init
yy_init = [df_init[df_init['subject_id'] == xlabel]['cost'].values[0]
           for xlabel in xticklabels]
plt.scatter(xticklabels, yy_init, marker='v', label='init')
# add random init
yy_random = [df_random[df_random['subject_id'] == xlabel]['cost'].values[0]
             for xlabel in xticklabels]
plt.scatter(xticklabels, yy_random, marker='v', label='random')
# add population cost
if df_pop_cost is not None:
    yy_pop = [df_pop_cost[df_pop_cost['subject_id'] == xlabel]['cost'].values[0]
           for xlabel in xticklabels]
    plt.scatter(xticklabels, yy_pop, marker='v', label='pop')

ax.legend()
plt.xticks(rotation=45) 
plt.xlabel('Subject ID')
plt.ylabel('Lasso cost')

if add_number:
    ax2 = ax.twinx()
    xx = list(range(len(subject_id_list)))
    yy = n_non_apnea
    ax2.plot(xx, yy, color="black", alpha=.6)
    ax2.set_ylim(0)
    ax2.set_ylabel('# non-apnea minutes')


# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(f'Recovery on group {group_id} ({group_des[group_id]})')
plt.tight_layout()
plt.savefig(f'recovery_{group_id}_{fit_on}.pdf', dpi=300)
plt.show()
plt.close()
# %%
