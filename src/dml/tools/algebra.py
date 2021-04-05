import numpy as np
from numpy.linalg import inv


def ols(X: np.ndarray, y: np.ndarray, fit_intercept=False):
    X, y = _1d_to_2d(X, y)
    if fit_intercept:
        intercept = [1] * X.shape[0]
        if not intercept in X.transpose().tolist():
            X = np.hstack((np.array(intercept).reshape((-1, 1)), X))
    xx = np.matmul(X.transpose(), X)
    xy = np.matmul(X.transpose(), y)
    return np.matmul(inv(xx), xy)

def iv_ols(X: np.ndarray, Z: np.ndarray, y: np.ndarray):
    X, Z, y = _1d_to_2d(X, Z, y)
    zx = np.matmul(Z.transpose(), X)
    zy = np.matmul(Z.transpose(), y)
    return np.matmul(inv(zx), zy)

def _1d_to_2d(*args: np.ndarray):
    new_args = []
    for arg in args:
        if len(arg.shape) == 1:
            arg = arg.reshape((-1, 1))
        new_args.append(arg)
    return tuple(new_args)