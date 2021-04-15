from dml.tools.utils import get_settings

from pathlib import Path

import pandas as pd
import numpy as np

settings = get_settings('lisa_sim.cfg')

simulations_folder = Path(__file__).parent.absolute() / 'simulations' / '{}dgp_{}k_{}n_{}splits_{}folds'.format(
    settings['model'],
    settings['k'],
    settings['n'],
    settings['n_splits'],
    settings['n_folds']
)

sims = list(simulations_folder.glob(pattern='*.csv'))

results = pd.DataFrame(columns=methods)

for file in sims:
    res = pd.read_csv(file, index_col=0)
    results = results.append(res)

results.reset_index(drop=True, inplace=True)
print(results)

# Summarize data
col_means = results.mean(axis=0)
col_medians = (abs(results - theta)).median(axis=0)
col_var = results.var(axis=0)
col_mse = ((results - theta)**2).mean(axis=0)
# hitrate
ols_rep = np.tile(results.to_numpy()[:, 0], (n_methods, 1)).transpose()
hitrate = [0] + (abs(results.to_numpy()[:, 1:] - theta) > abs(ols_rep - theta)).mean(axis=0).tolist()

summary = pd.DataFrame([col_means, col_medians, col_var, col_mse, pd.Series(hitrate, index=col_means.index)],
                       index=['Mean', 'MAE', 'Var', 'MSE', 'Hit rate'],
                       columns=methods).transpose()

# Convert to latex
latex_filename = Path(__file__).parent.absolute() / 'latex' / 'res_sim{}_k{}_n{}.txt'.format(
    len(sims), k, n
)
with open(latex_filename, 'w') as tex_file:
    tex_file.write(summary.to_latex())