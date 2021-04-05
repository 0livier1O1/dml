from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgbm


def _ml_Tree(X, y, tune=True):
    model = DecisionTreeRegressor()
    if tune:
        ccps = model.cost_complexity_pruning_path(X, y).ccp_alphas[:-1]
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        cv = GridSearchCV(estimator=model,
                          param_grid={'ccp_alpha': ccps.tolist()},
                          return_train_score=False,
                          scoring="neg_mean_squared_error",
                          cv=kfold)
        cv.fit(X, y)
        best_cp = cv.best_estimator_.get_params()['ccp_alpha']
        model.set_params(**{'ccp_alpha': best_cp})

    return model


def _ml_Forest(X, y, tune=False):
    model = RandomForestRegressor(n_estimators=200, n_jobs=1)
    return model


def _ml_Boosting(X, y, tune=False):
    model = lgbm.LGBMRegressor(num_leaves=900,
                               max_depth=50,
                               learning_rate=0.05,
                               n_estimators=200,
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
    pass
