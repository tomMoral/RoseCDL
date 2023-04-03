import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


df_cost = pd.read_csv('../scripts/df_cost.csv')
df_cost['subject_id'] = df_cost['subject_id'].apply(lambda x: x[4:])

# get sub-dataframe for D_init
d_init_index = df_cost[df_cost['dict_fit'] == 'D_init'].index
df_init = df_cost.loc[d_init_index.values]
df_init['cost'] = df_init['cost'].apply(
    lambda x: float(x.split(',')[-1][1:-1]))

df_cost.drop(d_init_index.values, inplace=True)
df_cost['cost'] = df_cost['cost'].astype(float)
df_cost['dict_fit'] = df_cost['dict_fit'].apply(lambda x: x[4:])

# get sub-dataframe for D_self
self_index = df_cost[df_cost['subject_id'] == df_cost['dict_fit']].index
df_self = df_cost.loc[self_index.values]

df_cost.drop(self_index.values, inplace=True)


df_boxplot = df_cost[df_cost['subject_id'] != df_cost['dict_fit']]
g = sns.boxplot(data=df_cost, x="subject_id", y="cost")

xticklabels = [t.get_text() for t in g.get_xticklabels()]

yy_self = [df_self[df_self['subject_id'] == xlabel]['cost'].values[0]
           for xlabel in xticklabels]
plt.scatter(xticklabels, yy_self, marker='*', label='self')

yy_init = [df_init[df_init['subject_id'] == xlabel]['cost'].values[0]
           for xlabel in xticklabels]
plt.scatter(xticklabels, yy_init, marker='v', label='init')

for i, df_cat in enumerate([df_cost_cat1, df_cost_cat2]):
    yy_cat = [df_cat[df_cat['subject_id'] == xlabel]['cost'].values[0]
              for xlabel in xticklabels]
    plt.scatter(xticklabels, yy_cat, marker='o', label=f'cat{i+1}', alpha=0.5)

yy_all = [df_cost_all[df_cost_all['subject_id'] == xlabel]['cost'].values[0]
          for xlabel in xticklabels]
plt.scatter(xticklabels, yy_all, marker='P', label='all', alpha=0.5)

plt.xticks(rotation=90)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
