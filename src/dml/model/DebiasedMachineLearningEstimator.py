from dml.tools.dgp import model1
from dml.tools.utils import get_path_to_file
from dml.tools.algebra import ols, iv_ols
from dml.model.mlestimators import _ml_Tree, _ml_Forest, _ml_Boosting, _ml_Neural_Network, _ml_Elastic_Net, _ml_Ridge

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator

from joblib import Parallel, delayed

import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import randint


class DML:

    def __init__(self, model_y, model_t, n_splits=100, n_folds=2, n_jobs=1, verbose=0, small_dml=False):
        self.model_y = model_y
        self.model_t = model_t
        self.n_splits = n_splits
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.small_dml = small_dml

        self.rmse_y = False
        self.rmse_t = False

        self._is_estimated = False

    def get_dml_rmse(self):
        if self._is_estimated:
            return self.rmse_y, self.rmse_t
        else:
            raise RuntimeError('Treatment effect not yet estimated.')

    def treatment_effect(self, X=None, y=None, t=None):
        self.X_ = X
        self.y_ = y
        self.t_ = t

        fold_seeds = randint.rvs(0, 1000, size=self.n_splits, random_state=0).tolist()
        treatment_effect= self._dml_estimation(fold_seeds)
        self._is_estimated = True

        return treatment_effect

    def _dml_estimation(self, fold_seeds):
        X_ = self.X_
        y_ = self.y_
        t_ = self.t_

        parallel = Parallel(n_jobs=self.n_jobs,
                            verbose=0, # self.verbose,
                            pre_dispatch='2 * n_jobs',
                            prefer='threads')
        dml_splits = parallel(delayed(self._parallel_estimate_single_split)(X_, y_, t_, i, fold_seeds[i])
                              for i in range(self.n_splits))
        dml_splits = np.array(dml_splits)

        treamtnet_effect = dml_splits[:, 0].mean()
        self.rmse_y, self.rmse_t = dml_splits[:, 1], dml_splits[:, 2]

        return treamtnet_effect

    def _parallel_estimate_single_split(self, X, y, t, split_idx, fold_seed):
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=fold_seed)

        rmse_y = 0
        rmse_t = 0

        TE = 0
        y_pool = []
        t_pool = []

        for fold, sets in enumerate(folds.split(X, y, t)):
            main, aux = sets
            X_main, y_main, t_main = X[main], y[main], t[main]
            X_aux, y_aux, t_aux = X[aux], y[aux], t[aux]

            y_resid, error_y = self._debiased_residuals(X_main, y_main, X_aux, y_aux, self.model_y)
            t_resid, error_t = self._debiased_residuals(X_main, t_main, X_aux, t_aux, self.model_t)

            y_pool += y_resid.tolist()
            t_pool += t_resid.tolist()

            theta_hat = ols(t_resid, y_resid).item()

            TE += theta_hat/self.n_folds
            rmse_y += error_y/self.n_folds
            rmse_t += error_t/self.n_folds

        if self.small_dml:
            TE = ols(np.array(t_pool), np.array(y_pool))

        if self.verbose == 1:
            print('Split {}/{} Completed'.format(split_idx + 1, self.n_splits))

        return TE, rmse_y, rmse_t

    def _debiased_residuals(self, X_main, y_main, X_aux, y_aux, method):
        # Fit on auxiliary sample
        method = method.replace(' ', '_')

        model = self._ml_estimator(X_aux, y_aux, method)

        # Predict on main sample
        y_pred = model.predict(X_main).flatten()

        # Compute Debiased Residuals || Y - E[Y|X] or D - E[D|X]
        residuals = y_main - y_pred

        # Compute MSE
        rmse = np.sqrt(((y_pred-y_aux)**2).mean(axis=0))

        return residuals, rmse

    def _ml_estimator(self, X, y, method: str) -> BaseEstimator:
        # TODO Set params per method
        # prepend _ml_ and call method on X, y

        if method == 'Lasso':
            model = _ml_Elastic_Net(X, y, l1_ratios=1)
        else:
            model = eval('_ml_' + method + '(X, y)')

        if method in ['Lasso', 'Ridge', 'Elastic_Net', 'Neural_Network']:
            scaler = MinMaxScaler() if method == 'Neural_Network' else StandardScaler()
            estimator = Pipeline([('scaler', scaler),
                                  ('model', model)])
        else:
            estimator = model

        estimator.fit(X, y)

        return estimator
