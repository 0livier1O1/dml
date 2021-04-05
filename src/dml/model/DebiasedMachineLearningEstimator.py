from dml.tools.utils import get_path_to_file
from dml.tools.algebra import ols, iv_ols
from dml.model.mlestimators import _ml_Tree, _ml_Forest, _ml_Boosting

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
from econml.dml import SparseLinearDML
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


class DML:

    def __init__(self, model_y, model_t, n_splits=100, n_folds=2, n_jobs=1, verbose=0):
        self.model_y = model_y
        self.model_t = model_t
        self.n_splits = n_splits
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __add__(self, other):
        # TODO: Allow for easy consolidation of multipled DML object with smaller scripts for jobs in multiple batches
        pass

    def treatment_effect(self, X=None, y=None, t=None):
        self.X_ = X
        self.y_ = y
        self.t_ = t

        np.random.seed(seed=1)
        fold_seeds = np.random.randint(0, 1000, size=self.n_splits).tolist()
        treatment_effect = self._dml_estimation(fold_seeds)
        return treatment_effect

    def _dml_estimation(self, fold_seeds):
        X_ = self.X_
        y_ = self.y_
        t_ = self.t_

        parallel = Parallel(n_jobs=self.n_jobs,
                            verbose=self.verbose,
                            pre_dispatch='2 * n_jobs',
                            prefer='threads')
        dml_splits = parallel(delayed(self._parallel_estimate_single_split)(X_, y_, t_, i, fold_seeds[i])
                              for i in range(self.n_splits))

        return np.array(dml_splits).mean()

    def _parallel_estimate_single_split(self, X, y, t, split_idx, fold_seed):
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=fold_seed)

        TE = 0

        for fold, sets in enumerate(folds.split(X, y, t)):
            main, aux = sets
            X_main, y_main, t_main = X[main], y[main], t[main]
            X_aux, y_aux, t_aux = X[aux], y[aux], t[aux]

            y_resid = self._debiased_residuals(X_main, y_main, X_aux, y_aux, self.model_y)
            t_resid = self._debiased_residuals(X_main, t_main, X_aux, t_aux, self.model_t)

            theta_hat = ols(t_resid, y_resid).item()

            TE += theta_hat

        return TE/self.n_folds

    def _debiased_residuals(self, X_main, y_main, X_aux, y_aux, method):
        """
        Function to partial out the nuissance function from the dependent variable
        Parameters
        ----------
        X_main : Numpy Matrix
            Matrix of confounders
        y_main : Numpy vector
            Vector of dependent variable
        X_aux : Numpy Matrix

        y_aux
        method

        Returns
        -------

        """
        # Fit on auxiliary sample
        model_y = self._ml_estimation(X_aux, y_aux, method) # TODO: Add settings
        # Predict on main sample
        y_pred = model_y.predict(X_main)
        # Compute Debiased Residuals || Y - E[Y|X] or D - E[D|X]
        residuals = y_main - y_pred
        return residuals

    def _ml_estimation(self, X, y, method) -> BaseEstimator:
        # TODO Set params per method
        # prepend _ml_ and call method on X, y
        model = eval('_ml_' + method + '(X, y)')

        if method in ['Lasso', 'Ridge']:
            l1_ratio = 1 if method == 'Lasso' else 0
            model.set_params(**{'l1_ratio': l1_ratio})

        return model.fit(X, y)


if __name__ == '__main__':
    start = time.time()
    lim = 150
    data = pd.read_csv(get_path_to_file("data_3.csv"), index_col=0)
    y = data.iloc[:, 0].to_numpy()
    t = data.iloc[:, 1].to_numpy()
    X = data.iloc[:, 2:].to_numpy()


    x = np.hstack((X, t.reshape(-1, 1)))
    res_ols = ols(x, y, fit_intercept=True)
    print('OLS')
    print(res_ols[-1])

    print('DML')
    dml = DML(model_y='Forest', model_t='Forest', n_folds=2, n_jobs=6, n_splits=100)
    res = dml.treatment_effect(X, y, t)
    print(res)
    print(time.time() - start)

    # OLS result

