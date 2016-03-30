import numpy as np


def _fallback_correlation(X, y, xy):
    '''Fallback if the cython operator get wrong results
    '''

    X = X - X.mean()
    y = y - y.mean()
    ls = len(X)
    lp = len(y)
    ny = np.sqrt(np.sum(y*y))
    nx = np.array([np.sqrt(np.sum((X[i:i+lp]-X[i:i+lp].mean())**2))
                   for i in range(ls-lp+1)])
    res = xy/(ny*nx)
    res[np.isnan(res)] = 0
    return res
