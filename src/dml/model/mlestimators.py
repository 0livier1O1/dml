from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import BatchNormalization
from keras.regularizers import l2

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import lightgbm as lgbm
import numpy as np

def _ml_Tree(X, y, tune=True):
    model = DecisionTreeRegressor(max_depth=20,
                                  min_samples_leaf=0.2,
                                  min_samples_split=round(30/3),
                                  ccp_alpha=0)
    if tune:
        min_samples_leaf = [0.2, 0.3, 0.4, 0.49]
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        cv = GridSearchCV(estimator=model,
                          param_grid={'min_samples_leaf': min_samples_leaf},
                          return_train_score=False,
                          scoring="neg_mean_squared_error",
                          cv=kfold)
        cv.fit(X, y)
        model = cv.best_estimator_
    return model

def _ml_Forest(X, y, tune=False):
    model = RandomForestRegressor(n_estimators=100, n_jobs=1) # 1.69, 2.01
    # model = lgbm.LGBMRegressor(boosting_type='rf',
    #                            n_estimators=100,
    #                            feature_fraction= 0.3, # np.sqrt(X.shape[1]) / X.shape[1],
    #                            subsample=0.632,
    #                            subsample_freq=1)
    return model

def _ml_Boosting(X, y, tune=False):
    model = lgbm.LGBMRegressor(num_leaves=900,
                               max_depth=50,
                               learning_rate=0.05,
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
    # standardization
    def single_hidden_layer():
        perceptron = Sequential()
        perceptron.add(Dense(10, input_dim=X.shape[1], kernel_initializer='normal',
                       kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        perceptron.add(Dense(1,  kernel_initializer='normal'))
        perceptron.compile(loss='mean_squared_error', optimizer='adam')
        return perceptron
    # model = KerasRegressor(build_fn=single_hidden_layer, nb_epoch=100, batch_size=30, verbose=0)
    model = MLPRegressor(hidden_layer_sizes=(10, ), solver='lbfgs',
                         max_iter=1200, learning_rate='adaptive', alpha=0.005)
    return model


def _ml_Elastic_Net(X, y, tune=True, l1_ratios=None):
    kfold = KFold(n_splits=4, shuffle=True, random_state=0)

    if tune:
        if l1_ratios is None:
            l1_ratios = [.1, .5, .7, .9, .95, .99]

        model = ElasticNetCV(cv=kfold,  n_alphas=70)

        model.set_params(**{'l1_ratio': l1_ratios})

    else:
        if l1_ratios is None:
            l1_ratios = 0.5

        model = ElasticNet(l1_ratio=l1_ratios)

    return model

def _ml_Ridge(X, y, tune=True):
    model = RidgeCV()
    return model