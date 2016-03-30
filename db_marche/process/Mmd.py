import numpy as np
import scipy.special as sp
from math import sqrt
import scipy.spatial.distance as dist
import matplotlib.mlab as mat


class Mmd(object):
    """ contains the temporary objects used to compute the mmd (in the Segmentation class)"""
    def __init__(self, exo, desc='CRZ|DAX'):
        """
        - exo: the exercice we wish to compute the segmentation of.
        - desc: str, optional (default: '')
            The names of the signals we want to use
            (see Exercise.get_signal to see the naming convention)
            A join can be made with '|'
        - _T: length of the signal
        - wsize: the size of the window we use to calculate the mmd.
        - hsize: size of the hop window. The mmd is computed every hsize point.
        - alpha: the size of the test "H0: p==q" and "H1: p!=q".
        - Dxx contains Dxx[i,j] = norm(x_i - x_j)**2
        - Dyy contains Dyy[i,j] = norm(y_i - y_j)**2
        - Dxy contains Dxy[i,j] = norm(x_i - y_j)**2
        - tt contains the point where the mmd at hand was computed.
        - UpToDate: once the mmd for the time tt is computed, UpToDate <- True.
            A change in exo, alpha, wsize, hsize, tt will set it to False.
        - mmd: the value of the mmd at hand
        - sigma: the asymptotical standard deviation of the mmd
        - epsilon: the lower bound of the asyptotical test of size alpha
            (epsilon = sqrt(2)*sp.erfinv(1-2*alpha)*sigma)
        - _C: the constant used to go from sigma to epsilon
        - MMD and EPSILON are arrays with the successive values of mmd and epsilon
        """
        self._exo = exo
        self._desc = desc
        self._T = exo.get_signal(desc.split('|')[0]).shape[1]
        self._wsize = int(2*100)  # TBD
        self._hsize = 1  # TBD
        self._alpha = 0.01
        self.Dxx = np.zeros((self._wsize, self._wsize))
        self.Dyy = np.zeros((self._wsize, self._wsize))
        self.Dxy = np.zeros((self._wsize, self._wsize))
        self._D = np.zeros((2*self._wsize, 2*self._wsize))
        self._tt = 0
        self.UpToDate = False
        self.mmd = 0
        self.MMD = list([])
        self.sigma = 0
        self.epsilon = 0
        self.EPSILON = list([])
        self._C = sqrt(2)*sp.erfinv(1-2*self.alpha)

    @property
    def wsize(self):
        return int(self._wsize)

    @wsize.setter
    def wsize(self, w):
        self._wsize = int(w)
        self.tt = 0
        self._D = np.zeros((2*w, 2*w))
        self.Dxx = np.zeros((w, w))
        self.Dyy = np.zeros((w, w))
        self.Dxy = np.zeros((w, w))
        self.UpToDate = False

    @property
    def hsize(self):
        return int(self._hsize)

    @hsize.setter
    def hsize(self, h):
        self._hsize = h
        self.tt = 0
        self.UpToDate = False

    @property
    def tt(self):
        return self._tt

    @tt.setter
    def tt(self, tt):
        self._tt = tt
        self.UpToDate = False

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, a):
        self._alpha = a
        self._C = self._C = sqrt(2)*sp.erfinv(1-2*self.alpha)
        self.UpToDate = False

    @property
    def desc(self):
        return self._desc

    @desc.setter
    def desc(self, desc):
        self._desc = desc
        self.UpToDate = False

    @property
    def exo(self):
        return self._exo

    @exo.setter
    def exo(self, exo):
        self._exo = exo
        self._T = exo.get_signal(self.desc).shape[1]
        self.UpToDate = False

    def NextIndex(self):
        """ this method is used to go from index tt to tt+hsize and compute the mmd.
        If max(t+hsize,t+wsize) is out of bounds, the UpToDate attribute is set to None."""
        if self.tt + max(self.hsize, self.wsize) + self.hsize < self._T:
            if self.tt == 0:
                self.tt = max(self.wsize, self.hsize)
            else:
                self.tt += self.hsize
            self.UpToDate = False
            self.set_mmd()
            self.MMD.append(self.mmd)
            self.EPSILON.append(self.epsilon)
            self.UpToDate = True
        else:
            self.UpToDate = None

    def set_distmat(self):
        """ this function returns 3 distance matrices of two given numpy arrays:
        - M1(i, j) = norm(a[i]-a[j])**2
        - M2(i, j) = norm(a[i]-b[j])**2
        - M3(i, j) = norm(b[i]-b[j])**2
        x and y must have same shape"""
        ind = range(-self.wsize, 0), range(0, self.wsize)
        s = np.column_stack(
            (self.exo.get_signal(desc=d)[0][np.concatenate(ind)+self.tt]
                for d in self.desc.split('|')))
        a, b = s[range(self.wsize)], s[range(self.wsize, 2*self.wsize)]
        if a.ndim == 1:
            a = a.reshape((a.shape[0], 1))
            b = b.reshape((a.shape[0], 1))
        vect = np.vstack((a, b))
        self._D = dist.squareform(dist.pdist(vect, 'euclidean'))
        n = a.shape[0]
        self.Dxx, self.Dxy, self.Dyy = (np.asarray(self._D[:n, :n]),
                                        np.asarray(self._D[n:, :n]),
                                        np.asarray(self._D[n:, n:]))

    def set_mmd(self, k=np.exp):
        """ This function calculates the mmd using the inter class distance matrices (Dxx,Dxy,Dyy).
        The kernel is gaussian and the bandwidth is dynamically fitted. """
        # computing the distance matrices
        self.set_distmat()
        # setting the kernel bandwidth
        # the indices for the upper-triangle of an (n, n) array, without the diagonal
        ind = np.triu_indices(self.Dxx.shape[0], k=1)
    #    to compute the median of the inter vector distance, we need:
    #    the upper triangle of Dxx and Dyy and the whole matrix Dxy except the diagonal.
        dxy = np.ma.array(self.Dxy, mask=False)
        dxy.mask[np.diag_indices(self.Dxx.shape[0])] = True
        vect = np.ma.concatenate((self.Dxx[ind], dxy.flatten(), self.Dyy[ind]))
        mdist = np.ma.median(vect)
        bwidth = sqrt(mdist/2.0)
    #        Kxx[i,j] = exp(-norm(x_i - x_j)**2 / (2*bwidth**2))
        C = -1./(2.*bwidth*bwidth)
        Kxx = k(self.Dxx*C)
        Kyy = k(self.Dyy*C)
        Kxy = k(self.Dxy*C)
        # we put the diag to zero because we do not want to sum those terms
        np.fill_diagonal(Kxx, 0.)
        np.fill_diagonal(Kxy, 0.)
        np.fill_diagonal(Kyy, 0.)
    #        h(z_i,z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(y_i, x_j) for i!=j (0 if i==j)
        h = Kxx + Kyy - Kxy - np.transpose(Kxy)
    #        we calculate the mmd and its 99% reject area
        m = self.Dxx.shape[0]
        self.mmd = np.sum(h)
        self.mmd /= m*(m-1)*1.
        self.sigma = sum(sum(row)*sum(row) for row in h)
        del h
        self.sigma *= 4.
        self.sigma /= m*m*(m-1)*(m-1)
        self.sigma -= 4.*self.mmd*self.mmd/m
        self.sigma = sqrt(self.sigma)
        self.epsilon = self._C*self.sigma
        return self.mmd, self.epsilon

    def compute_mmd(self):
        self.MMD = list([])
        self.EPSILON = list([])
        while self.UpToDate is not None:
            self.NextIndex()
        return self.MMD, self.EPSILON

    def find_max(self, level=None):
        """ find the global maximums between two points that are above a given level.
        s is the univariate signal (numpy array)."""
        s = np.array(self.MMD)
        if level is None:
            level = max(self.EPSILON)
        elif level < np.min(s):
            return np.argmax(s)

        # we first get the local maxima of the signal
        # vector of booleans: True if this is the index of a local max of the signal, false if not.
        ind = np.r_[True, s[1:] > s[:-1]] & np.r_[s[:-1] > s[1:], True]
        # we only keep the greatest local max between:
        # - a point where the signal goes above the threshold
        # - a point where the signal goes below the threshold.
        m_ind = ind.nonzero()[0]  # indices of the local max
        del ind

        def gen():
            """ this generator yields the start and end point of the signal above the threshold."""
            # ind: the indexes where the signal goes below or above the threshold
            ind = mat.find(np.diff(s > level))+1
            IsIn = True
            if s[0] > level:
                yield 0, ind[0]
                IsIn = not IsIn
            for i, val in enumerate(ind[:-1]):
                if IsIn:
                    yield val, ind[i+1]
                IsIn = not IsIn
            if s[-1] > level:
                yield ind[-1], s.shape[0]

        res = list()
        for i, j in gen():
            sub_s = s[i:j]
            ind = np.argmax(sub_s)+i
            # we check if the max we have is at least a local max for the signal s.
            if ind in m_ind:
                res.append(ind)
        return np.asarray(res)
