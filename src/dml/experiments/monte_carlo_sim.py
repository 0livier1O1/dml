from dml.model.DebiasedMachineLearningEstimator import DML
from dml.tools.dgp import model1
from dml.tools.algebra import ols

from pathlib import Path

import time
import numpy as np
import pandas as pd


M = 100 # Number of simulations
k = 90
n = 100
theta = 2
linear = False
methods = ['OLS', 'Boosting']
n_methods = len(methods) - 1

results = {method: [] for method in methods}

start = time.time()

for i in range(M):
    np.random.seed(i + 1)
    print('Iteration {}'.format(i + 1))
    y, t, X = model1(k, n, linear, T=theta)

    for method in methods:
        if method == 'OLS':
            print('Fitting OLS')

            x = np.hstack((X, t.reshape(-1, 1)))
            res_ols = ols(x, y, fit_intercept=True)
            results['OLS'].append(res_ols[-1].item())

        else:
            print('Fitting DML with {}'.format(method))

            model = DML(model_y=method, model_t=method, n_folds=2, n_jobs=4, n_splits=100, verbose=0, small_dml=True)
            res_dml = model.treatment_effect(X, y, t)
            results[method].append(res_dml)

print(time.time() - start)
results = pd.DataFrame(results)
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

print(summary)

# Convert to latex
if len(methods) == 2:
    dgp_type = "linear" if linear else "nonlinear"
    latex_filename = Path(__file__).parent.absolute() / 'latex' / 'res_{}_dgp_sim{}_k{}_n{}_{}.txt'.format(
        dgp_type, M, k, n, methods[-1].replace(' ', '')
    )

    latex_filename.parent.mkdir(parents=True, exist_ok=True)

    with open(latex_filename, 'w') as tex_file:
        tex_file.write(summary.to_latex())
