from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

from warnings import filterwarnings

from pprint import pprint as pp

filterwarnings(action='ignore', category=ConvergenceWarning)

# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# from keras.layers import BatchNormalization
# from keras.regularizers import l2
#
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import lightgbm as lgbm
import numpy as np

def _ml_Tree(X, y, tune=True):
    model = DecisionTreeRegressor(max_depth=25,
                                  min_samples_leaf=0.2,
                                  min_samples_split=round(20/3),
                                  ccp_alpha=1)
    if tune:
        min_samples_leaf = [0.1, 0.2, 0.3, 0.4, 0.49]
        kfold = KFold(n_splits=10, shuffle=True, random_state=0)
        cv = GridSearchCV(estimator=model,
                          # param_grid={'min_samples_leaf': min_samples_leaf},
                          param_grid={'min_samples_leaf': min_samples_leaf},
                          return_train_score=False,
                          scoring="neg_mean_squared_error",
                          cv=kfold)
        cv.fit(X, y)
        model = cv.best_estimator_
    return model

def _ml_Forest(X, y, tune=False):
    model = RandomForestRegressor(n_estimators=100, n_jobs=1, min_samples_leaf=1) # 1.69, 2.01
    return model

def _ml_Boosting(X, y, tune=False):
    model = lgbm.LGBMRegressor(num_leaves=700,
                               max_depth=40,
                               learning_rate=0.03,
                               n_estimators=100,
                               reg_lambda=1000)
    if tune:
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        param_dist = {
            "max_depth": [25, 75],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [300, 900],
            "n_estimators": [200],
            "reg_lambda": [1000]
        }
        cv = GridSearchCV(model,
                          param_grid=param_dist,
                          cv=3,
                          scoring="neg_mean_squared_error")
        cv.fit(X, y)
        model.set_params(**cv.best_estimator_.get_params())

    return model

def _ml_Neural_Network(X, y, tune=True):
    model = MLPRegressor(hidden_layer_sizes=(12, ), solver='lbfgs',
                         max_iter=1200, learning_rate='adaptive', alpha=0.1)
    return model

def _ml_Elastic_Net(X, y, tune=True, l1_ratios=None):
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    # TODO Standardize y to see if it changes something
    if tune:
        if l1_ratios is None:
            l1_ratios = [0.1, 0.25, 0.5, 0.75, 1]

        # TODO Less Alphas
        model = ElasticNetCV(cv=kfold,  n_alphas=100)

        model.set_params(**{'l1_ratio': l1_ratios})

    else:
        if l1_ratios is None:
            l1_ratios = 0.5

        model = ElasticNet(l1_ratio=l1_ratios)

    return model

def _ml_Ridge(X, y, tune=True):
    model = RidgeCV()
    return model