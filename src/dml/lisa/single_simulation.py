# When runing from LISA
import sys
sys.path.append(0, '/home/omulkin')

from dml.model.DebiasedMachineLearningEstimator import DML
from dml.tools.dgp import *
from dml.tools.algebra import ols
from dml.tools.utils import get_settings

from pathlib import Path

import time
import numpy as np
import pandas as pd

settings = get_settings('lisa_sim.cfg')

methods = settings['methods']
n_methods = len(methods) - 1
results = {method: [] for method in methods}

start = time.time()

dgp = get_dgp(model=settings['model'])

y, t, X = dgp(k=settings['k'], n=settings['n'], linear=settings['linear'], T=settings['treatment_coef'])

for method in methods:
    if method == 'OLS':
        print('Fitting OLS')

        x = np.hstack((X, t.reshape(-1, 1)))
        res_ols = ols(x, y, fit_intercept=True)
        results['OLS'].append(res_ols[-1].item())

    else:
        print('Fitting DML with {}'.format(method))

        model = DML(model_y=method,
                    model_t=method,
                    n_folds=settings['n_folds'],
                    n_jobs=-1,
                    n_splits=settings['n_splits'],
                    verbose=0,
                    small_dml=settings['DML2'])

        res_dml = model.treatment_effect(X, y, t)
        results[method].append(res_dml)

print(time.time() - start)
results = pd.DataFrame(results)

simulations_folder = Path(__file__).parent.absolute() / 'simulations' / '{}dgp_{}k_{}n_{}splits_{}folds'.format(
    settings['model'],
    settings['k'],
    settings['n'],
    settings['n_splits'],
    settings['n_folds']
)

simulations_folder.mkdir(parents=True, exist_ok=True)

filename = simulations_folder / 'PLR_{}.csv'.format(np.random.randint(0, 1000000))
results.to_csv(filename)
