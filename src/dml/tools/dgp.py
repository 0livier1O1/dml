import random

import numpy as np
import pandas as pd

from numpy import exp, log, abs
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

def dgp(k=20, n=100, linear=False, T=2, corr = 0.2):
    cov = np.fromfunction(lambda i, j: corr**np.abs(i - j), (k, k))
    X = np.random.multivariate_normal(np.zeros(k), cov, size=n)

    if linear:
        x_g = X[:, 1:10]
        x_m = X[:, 7:13]
    else:
        x_g = np.column_stack((
            exp(X[:, 0]) * exp(X[:, 1]),
            exp(X[:, 2]) * exp(X[:, 3]),
            X[:, 4] * X[:, 5], X[:, 6] * X[:, 7],
            X[:, 8]**2, X[:, 9]**2,
            log(abs(X[:, 10] + 1)) * log(abs(X[:, 11] + 1)),
            log(abs(X[:, 3] + 1)), 1/X[:, 4], norm.pdf(X[:, 12:14])
        ))
        x_m = np.column_stack((
            exp(X[:, 0]) * exp(X[:, 2]),
            X[:, 13] * X[:, 14], X[:, 15] * X[:, 16],
            X[:, 10]**2, log(abs(X[:, 8] + 1)) * log(abs(X[:, 17] + 1)),
            1/X[:, 16], log(abs(X[:, 2] + 1))
        ))

    coefs = [-5, -3, -1, 1, 3, 5]

    c_g = np.random.choice(coefs, size=x_g.shape[1], replace=True).reshape((-1, 1))
    c_m = np.random.choice(coefs, size=x_m.shape[1], replace=True).reshape((-1, 1))

    error = np.random.multivariate_normal(np.zeros(2), np.identity(2), size=n)

    scaler = MinMaxScaler((-10, 10))

    gamma_PLR = scaler.fit_transform(x_m @ c_m) + error[:, 0].reshape((-1, 1))
    y_PLR = T * gamma_PLR + scaler.fit_transform(x_g @ c_g) + error[:, 1].reshape((-1, 1))

    return y_PLR.flatten(), gamma_PLR.flatten(), X


