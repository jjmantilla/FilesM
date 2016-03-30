# -*- coding: utf-8 -*-
import numpy as np
from .. import DATA_DIR

# Load a cython correlation function, with compatibility
# for system wihtout cython
try:
    from .cython_step.aux_step_detection import _correlation
except ImportError:
    print('WARNING - no cython module found. Will use a slower '
          'correlation function. Please consider installing cython'
          'for better performances')
    from .util import _fallback_correlation as _correlation
from .util import _fallback_correlation


class StepDetection(object):
    """Algorithm to compute the step detection for an exercise

    Parameters
    ----------
    lmbd: float, optional (default: .8)
        minimal correlation with a pattern for the step detection
    mu: float, optional (defualt: .2)
        minimal movement for the step detection (avoid noise detection)
    patterns: list of Pattern, optional (default: None)
        Library of patterns for the step detection algorithm.
        If none is provided, the algorithm will use the default one contained
        in DATA_DIR/DB_steps.npy
    """
    def __init__(self, lmbd=.8, mu=.2, patterns=None):
        self.lmbd = lmbd
        self.mu = mu

        # Load the default library if None is provided
        if not patterns:
            import os.path as osp
            db_pattern = osp.join(DATA_DIR, 'DB_steps.npy')
            patterns = np.load(db_pattern)
        self.patterns = patterns

    def compute_steps(self, exo):
        ''' Compute the step detection

        Return
        ------
        steps : list
            list of the steps times for each foot
        steps_label : list
            list of detection information for each foot

        Parameters
        ----------
        exo: Exercise
            Exercise with the data of the acquisition
        '''

        l_steps = [[], []]
        res = [[], []]
        steps, steps_label = [], []
        T = exo.data_sensor.shape[1]
        for foot in range(2):
            sigAV = exo.data_earth[6*foot+2]
            sigAZ = exo.data_sensor[6*foot+2]
            sigRY = exo.data_sensor[6*foot+4]
            sigAV = sigAV - sigAV.mean()
            sigAZ = sigAZ - sigAZ.mean()
            sigRY = sigRY - sigRY.mean()
            L = []
            for n_pattern, pattern in enumerate(self.patterns):
                if pattern.coord == 'AV':
                    sig = sigAV
                elif pattern.coord == 'AZ':
                    sig = sigAZ
                else:
                    sig = sigRY

                # Compute the similarity distance
                try:
                    xy = np.convolve(sig, pattern.data[::-1],
                                     mode='valid')
                    correlation = _correlation(sig, pattern.data, xy)
                except AssertionError as e:
                    print('WARNING: Fallback to python correlation called')
                    print('\t\t', e.args[0])
                    correlation = _fallback_correlation(sig, pattern.data,
                                                        xy)

                # Extract the ordered maxima
                i0 = np.where((correlation[:-2] < correlation[1:-1]) &
                              (correlation[1:-1] > correlation[2:]))[0]+1

                L += [(correlation[i], i, n_pattern) for i in i0
                      if correlation[i] >= self.lmbd]

            used = np.zeros(T, dtype=bool)
            L.sort()
            for a, position, n_pattern in L[::-1]:
                l_pattern = len(self.patterns[n_pattern].data)
                if not (used[position:position+l_pattern]).any():
                    used[position:position+l_pattern] = True
                    l_steps[foot] += [(position,
                                       position+l_pattern,
                                       n_pattern, a)]
                    res += [[position, position+l_pattern]]

            steps += [[]]
            for step in l_steps[foot]:
                pattern = self.patterns[step[2]]
                th_noise = self.mu*np.std(pattern.data)
                if pattern.coord == 'AV':
                    sig = sigAV
                elif pattern.coord == 'AZ':
                    sig = sigAZ
                else:
                    sig = sigRY
                if np.std(sig[step[0]:step[1]]) > th_noise:
                    steps[foot] += [[step[0], step[1], step[2]]]
            steps[foot].sort()
            steps_label += [[v[2] for v in steps[foot]]]
            steps[foot] = [[v[0], v[1]] for v in steps[foot]]

        return steps, steps_label
