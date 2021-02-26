from dml.tools.utils import get_path_to_file

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet

from joblib import Parallel, delayed

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

    def fit(self, X=None, y=None, g=None):
        self.X_ = X
        self.y_ = y
        self.g_ = g

        fold_seeds = np.random.randint(0, 1000, size=self.n_splits).tolist()
        for method in self.methods:
            a = self._causal_estimation(method, fold_seeds)
        return a

    def _causal_estimation(self, method, fold_seeds):
        # Compute folds here
        X_ = self.X_
        y_ = self.y_
        g_ = self.g_

        parallel = Parallel(n_jobs=self.n_jobs,
                            verbose=self.verbose,
                            pre_dispatch='2 * n_jobs',
                            prefer='threads')
        # Some ML methods are very slow: If each split has to wait for ML methods to finish, some cores will be idle and
        # this will create a
        dml_splits = parallel(delayed(self._parallel_estimate_single_split)(X_, y_, g_, i, method, fold_seeds[i])
                              for i in range(self.n_splits))

        return dml_splits

    def _parallel_estimate_single_split(self, X, y, g, split_idx, method, fold_seed):
        # TODO Check that the same folds splits are used accross methods
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=fold_seed)

        for fold, sets in enumerate(folds.split(X, y, g)):
            main_sample, aux_sample = sets
            X_main, y_main, g_main = X[main_sample], y[main_sample], g[main_sample]
            X_aux, y_aux, g_aux = X[aux_sample], y[aux_sample], g[aux_sample]

            model_y = self._ml_estimation(X_main, y_main, method)  # TODO: Add settings
            model_g = self._ml_estimation(X_main, g_main, method)  # TODO: Add settings

        return split_idx

    def _ml_estimation(self, X, y, method):
        ml_regressors = {
            'Tree': DecisionTreeRegressor,
            'Forest': RandomForestRegressor,
            'Lasso': ElasticNet,
            'Ridge': ElasticNet,
            'Elnet': ElasticNet,
            'Boosting': AdaBoostRegressor
        }
        estimator = ml_regressors[method]()
        if method in ['Lasso', 'Ridge']:
            l1_ratio = 1 if method == 'Lasso' else 0
            estimator.set_params(**{'l1_ratio': l1_ratio})

        return estimator.fit(X, y)


if __name__ == '__main__':
    lim = 150
    data = pd.read_csv(get_path_to_file("data_2.csv"), index_col=0)
    y = data.iloc[:lim, 0].to_numpy()
    g = data.iloc[:lim, 1].to_numpy()
    X = data.iloc[:lim, 2:].to_numpy()

    dml = DebiasedMachineLearningEstimator(methods=['Tree'], n_folds=2, n_jobs=1, n_splits=10)
    dml.fit(X, y, g)
