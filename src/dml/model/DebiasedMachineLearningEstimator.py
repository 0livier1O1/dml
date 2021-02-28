from dml.tools.utils import get_path_to_file
from dml.tools.algebra import ols, iv_ols
from dml.model.mlestimators import _ml_Tree, _ml_Forest

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet

from joblib import Parallel, delayed

import time
import numpy as np
import pandas as pd
from pprint import pprint as pp


class DebiasedMachineLearningEstimator:

    def __init__(self, n_splits=100, methods=None, n_folds=2, n_jobs=1, verbose=0):
        self.n_splits = n_splits
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.methods = methods
        self.verbose = verbose

    def __add__(self, other):
        # TODO: Allow for easy consolidation of multipled DML object with smaller scripts for jobs in multiple batches
        pass

    def fit(self, X=None, y=None, g=None):
        self.X_ = X
        self.y_ = y
        self.g_ = g

        fold_seeds = np.random.randint(0, 1000, size=self.n_splits).tolist()
        res = []
        for method in self.methods:
             res.append(self._dml_with_method(method, fold_seeds))
        return res

    def _dml_with_method(self, method, fold_seeds):
        X_ = self.X_
        y_ = self.y_
        g_ = self.g_

        parallel = Parallel(n_jobs=self.n_jobs,
                            verbose=self.verbose,
                            pre_dispatch='2 * n_jobs',
                            prefer='threads')
        # Some ML methods are very slow: If each split has to wait for ML methods to finish, some cores will be idle and
        # this will create a bottleneck: Better parallise accross split one method at at time
        dml_splits = parallel(delayed(self._parallel_estimate_single_split)(X_, y_, g_, i, method, fold_seeds[i])
                              for i in range(self.n_splits))

        return np.array(dml_splits).mean()

    def _parallel_estimate_single_split(self, X, y, g, split_idx, method, fold_seed):
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=fold_seed)

        TE = 0

        for fold, sets in enumerate(folds.split(X, y, g)):
            main_sample, aux_sample = sets
            X_main, y_main, g_main = X[main_sample], y[main_sample], g[main_sample]
            X_aux, y_aux, g_aux = X[aux_sample], y[aux_sample], g[aux_sample]

            y_resid = self._debiased_residuals(X_main, y_main, X_aux, y_aux, method)
            d_resid = self._debiased_residuals(X_main, g_main, X_aux, g_aux, method)

            # theta_hat = ols(d_resid, y_resid).item()  # The paper seem to say: We need IV with D_i => Not observed right?
            theta_hat = iv_ols(g_main.reshape((-1, 1)), d_resid, y_resid).item()

            TE += theta_hat/self.n_folds

        return TE

    def _debiased_residuals(self, X_main, y_main, X_aux, y_aux, method):
        """
        Function to compute the debiased residuals of the Partially Linear Regression model
        Parameters
        ----------
        X_main : Numpy Matrix
            Matrix of confounders
        y_main : Numpy
        X_aux
        y_aux
        method

        Returns
        -------

        """
        # Fit on auxiliary sample
        model_y = self._ml_estimation(X_aux, y_aux, method) # TODO: Add settings
        # Predict on main sample
        y_pred = model_y.predict(X_main)
        # Compute Debiased Residuals
        residuals = y_main - y_pred
        return residuals

    def _ml_estimation(self, X, y, method) -> BaseEstimator:
        # TODO Set params per method
        # prepend _ml_ and call method on X, y
        model = eval('_ml_' + method + '(X, y)')

        if method in ['Lasso', 'Ridge']:
            l1_ratio = 1 if method == 'Lasso' else 0
            model.set_params(**{'l1_ratio': l1_ratio})

        return model


if __name__ == '__main__':
    start = time.time()
    lim = 150
    data = pd.read_csv(get_path_to_file("data_1.csv"), index_col=0)
    y = data.iloc[:, 0].to_numpy()
    g = data.iloc[:, 1].to_numpy()
    X = data.iloc[:, 2:].to_numpy()

    dml = DebiasedMachineLearningEstimator(methods=['Tree', 'Forest'], n_folds=2, n_jobs=6, n_splits=100)
    res = dml.fit(X, y, g)
    print(res)
    print(time.time() - start)
