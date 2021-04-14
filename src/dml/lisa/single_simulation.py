from dml.model.DebiasedMachineLearningEstimator import DML
from dml.tools.dgp import dgp
from dml.tools.algebra import ols

from pathlib import Path

import time
import numpy as np
import pandas as pd

k = 90
n = 100
theta = 2
splits = 100
methods = ['OLS', 'Tree', 'Lasso']
n_methods = len(methods) - 1
results = {method: [] for method in methods}

start = time.time()

y, t, X = dgp(k, n, linear=False, T=theta)

for method in methods:
    if method == 'OLS':
        print('Fitting OLS')

        x = np.hstack((X, t.reshape(-1, 1)))
        res_ols = ols(x, y, fit_intercept=True)
        results['OLS'].append(res_ols[-1].item())

    else:
        print('Fitting DML with {}'.format(method))

        model = DML(model_y=method, model_t=method, n_folds=3, n_jobs=-1, n_splits=splits, verbose=0, small_dml=True)
        res_dml = model.treatment_effect(X, y, t)
        results[method].append(res_dml)

print(time.time() - start)
results = pd.DataFrame(results)

filename = Path(__file__).parent.absolute() / 'simulations' / 'PLR_{}.csv'.format(np.random.randint(0, 1000000))
results.to_csv(filename)
