from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def _ml_Tree(X, y, tune=True):
    model = DecisionTreeRegressor()
    if tune:
        ccps = model.cost_complexity_pruning_path(X, y).ccp_alphas[:-1]
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        cv = GridSearchCV(estimator=model,
                          param_grid={'ccp_alpha': ccps.tolist()},
                          return_train_score=False,
                          cv=kfold)
        cv.fit(X, y)
        best_cp = cv.best_estimator_.get_params()['ccp_alpha']
        model.set_params(**{'ccp_alpha': best_cp})

    return model.fit(X, y)


def _ml_Forest(X, y, tune=False):
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
    return model.fit(X, y)
