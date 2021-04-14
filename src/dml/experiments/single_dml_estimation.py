from dml.model.DebiasedMachineLearningEstimator import DML

import time
import numpy as np

start = time.time()
y, t, X = dgp(k=90, n=100)

dml = DML(model_y='Neural Network', model_t='Neural Network', n_folds=2, n_jobs=7, n_splits=100)
res = dml.treatment_effect(X, y, t)
print('DML')
print(res)
print(time.time() - start)

x = np.hstack((X, t.reshape(-1, 1)))
res_ols = ols(x, y, fit_intercept=True)
print('OLS')
print(res_ols[-1])