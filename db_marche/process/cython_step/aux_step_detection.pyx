import numpy as np
cimport numpy as np

from libc.math cimport sqrt, abs

cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _correlation(np.ndarray[DTYPE_t, ndim=1] X,
                 np.ndarray[DTYPE_t, ndim=1] pattern,
                 np.ndarray[DTYPE_t, ndim=1] xy):
    assert X.dtype == DTYPE
    cdef unsigned int lp = pattern.shape[0]
    cdef unsigned int N = X.shape[0]-lp+1
    cdef unsigned int i, l

    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(N, dtype=DTYPE)

    cdef double norm_pattern = 0, sum_pattern = 0
    cdef double sum_x = 0, sum_x2 = 0, xi, norm_x

    for i in range(lp):
        xi = X[i]
        sum_x += xi
        sum_x2 += xi*xi
        pi = pattern[i]
        sum_pattern += pi
        norm_pattern += pi*pi
    norm_pattern = norm_pattern-sum_pattern*sum_pattern/lp
    norm_x = sum_x2-sum_x*sum_x/lp
    if abs(norm_pattern) < 1e-7:
        print('WARNING: The library got a zero pattern!')
        return res
    if abs(norm_x) >= 1e-7:
        res[0] = ((xy[0]-(sum_x*sum_pattern)/lp) /
                  sqrt(norm_pattern*norm_x))

    # Linear computation of the cross correlation
    for i in range(N-1):
        # Update the sum for the signal
        xi = X[i]
        xt = X[i+lp]
        sum_x += xt-xi
        sum_x2 += xt*xt-xi*xi
        norm_x = sum_x2 - sum_x*sum_x/lp

        # Verify that the computational error is not to high
        if abs(norm_x) < 1e-7:
            res[i+1] = 0
        else:
            res[i+1] = ((xy[i+1]-(sum_x*sum_pattern)/lp) /
                        sqrt(norm_pattern*(sum_x2-sum_x*sum_x/lp)))

        # Clip to 1 when small error
        if res[i+1] >= 1 and np.isclose(res[i+1], 1):
            res[i+1] = 1.0
        if res[i+1] <= -1 and np.isclose(res[i+1], -1):
            res[i+1] = -1.0

        assert abs(res[i+1]) <= 1, (
            'Fail to compute correlation: {}'
            ''.format(res[i+1]))
    return res

