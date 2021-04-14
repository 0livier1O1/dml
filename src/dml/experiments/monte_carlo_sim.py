from dml.model.DebiasedMachineLearningEstimator import DML
from dml.tools.dgp import dgp
from dml.tools.algebra import ols

import time
import numpy as np
import pandas as pd

M = 100 # Number of simulations
k = 90
n = 100
theta = 2
methods = ['OLS', 'Tree', 'Lasso', 'Elastic Net', 'Boosting', 'Neural Network']
n_methods = len(methods) - 1

results = {method: [] for method in methods}
np.random.seed(100)

start = time.time()

for i in range(M):
    print('Iteration {}'.format(i))
    y, t, X = dgp(k, n, linear=False, T=theta)

    for method in methods:
        if method == 'OLS':
            print('Fitting OLS')

            x = np.hstack((X, t.reshape(-1, 1)))
            res_ols = ols(x, y, fit_intercept=True)
            results['OLS'].append(res_ols[-1].item())

        else:
            print('Fitting DML with {}'.format(method))

            model = DML(model_y=method, model_t=method, n_folds=3, n_jobs=6, n_splits=100, verbose=0, small_dml=True)
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

table = pd.DataFrame([col_means, col_medians, col_var, col_mse, pd.Series(hitrate, index=col_means.index)],
                     index=['Mean', 'MAE', 'Var', 'MSE', 'Hit rate'],
                     columns=methods).transpose()
print(table)
