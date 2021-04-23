from dml.model.DebiasedMachineLearningEstimator import DML

from dml.tools.dgp import model1
from dml.tools.algebra import ols

from pprint import pprint as pp

import time
import numpy as np
import pandas as pd

np.random.seed(2)

start = time.time()
y, t, X = model1(k=90, n=100, linear=False)

model = 'Neural Network'

dml = DML(model_y=model, model_t=model, n_folds=2, n_jobs=4, n_splits=100, small_dml=True)
res = dml.treatment_effect(X, y, t)

rmse_y, rmse_t = dml.get_dml_rmse()
rmse = pd.DataFrame([rmse_y, rmse_t], index=['MSE[Y|X]', 'MSE[D|X]']).transpose()
pp(rmse.describe())
print('DML')
print(res)
print(time.time() - start)

x = np.hstack((X, t.reshape(-1, 1)))
res_ols = ols(x, y, fit_intercept=True)
print('OLS')
print(res_ols[-1])