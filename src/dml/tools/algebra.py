import numpy as np
from numpy.linalg import inv


def ols(X: np.ndarray, y: np.ndarray):
    X, y = _1d_to_2d(X, y)
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