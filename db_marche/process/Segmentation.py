# -*- coding: utf-8 -*-
import numpy as np
import warnings
from db_marche.utils2 import exceptions as exc
# labels
still, uturn, walk = "still", "uturn", "walk"




def catch_np_warning(msg, exception=FloatingPointError):
    """ Decorate functions where there might be an EmptyStep exception raised.
    msg: string. Message to print in the logging warning
    exception: Exception. Exception to catch.

    Raises an EmptyStep exception in a logging warning.
    """
    assert issubclass(exception, Exception)

    def wrapper(func):
        def wrapped(e):
            old_settings = np.geterr()
            np.seterr(invalid="raise")
            try:
                return func(e)
            except exception:
                warnings.warn(msg + " " + e.fname, exc.EmptyStep)
            np.seterr(**old_settings)
        return wrapped
    return wrapper


def get_pas(e):
    """
    Returns a list: [[start, end],...] (start and end of each step, left and
    right foot)
    """
    #print(e.steps_annotation)
    #pas = e.steps_annotation[0] + e.steps_annotation[1]

    #print(e.steps)
    pas = e.steps[0] + e.steps[1]
    pas.sort(key=lambda x: x[0])
    return pas


@catch_np_warning("Problem when computing step features:", FloatingPointError)
def get_signal_of_feat(e):
    feat = lambda z: abs(np.mean(z))
    signal = e.get_signal("rCRX").T.flatten()
    pas = get_pas(e)
    features = np.array([feat(signal[start:end]) for (start, end) in pas])
    features -= min(features)
    features /= max(features)
    return features

class Segmentation(object):

    def __init__(self, e,level=None, maxiter=3, n_to_end=3, n_to_uturn=2,
                 toplevel=0.4):
        self.e = e
        self.pas = get_pas(self.e)
        self.labels = [walk] * len(self.pas)
        self.feats = get_signal_of_feat(self.e)
        self.level = level
        self._maxiter = maxiter
        self._n_to_end = n_to_end
        self._n_to_uturn = n_to_uturn
        self.toplevel = toplevel
        self.seg = [0, 0, 0, 0]

    def _is_above(self, consider=[walk]):
        """
        Returns a list of 0 and 1. Same length as self.pas.
        res[i] if the i-th step is above the level and no step between the i-th
        step and the argmax is below.
        Only steps whose label is in "consider" are considered.
        """
        try:
            assert self.level is not None
        except:
            print("self.level is not set (in __init__)")  # pragma: no cover
            raise Exception  # pragma: no cover

        signal = np.array(self.feats)
        # we keep only the steps whose label is in "consider"
        ind = np.logical_not(np.in1d(self.labels, consider))
        signal[ind] = - np.inf
        assert len(signal) == len(self.feats)
        length = len(signal)

        ind_max = np.nanargmax(signal)
        if signal[ind_max] < self.level:  # No step fit the criterion.
            return [0] * length
        if signal[ind_max] < self.toplevel:  # No step above the top level
            return [0] * length

        # on cherche le dernier passage au dessus de "level" et avant "ind_max"
        ind = [j for ((i, v), (j, w)) in
               zip(enumerate(signal[:-1]), enumerate(signal[1:], start=1)) if
               v < self.level < w]
        try:
            # dernier avant le demi-tour
            start = max(i for i in ind if i <= ind_max)
        except:  # il n'y pas eu de passage par le niveau "level" avant
            # "ind_max"
            start = 0  # la phase commence avec le premier pas.

        # on cherche le dernier pas au dessus de "level" et après "ind_max"
        ind = [i for ((i, v), (j, w)) in
               zip(enumerate(self.feats[:-1]),
                   enumerate(self.feats[1:], start=1)) if
               v > self.level > w]
        try:
            # dernier pas de la phase
            end = min(i for i in ind if i >= ind_max)
        except:
            end = length - 1

        res = np.zeros(length)
        res[start:(end + 1)] = 1
        # res = [0] * (start + 1) + [1] * (end - start) +\
        #     [0] * (length - end - 1)
        # assert len(res) == length
        return res

    def _set_label(self, binvect, label):
        # binvect must be an iterable of 0 and 1
        assert len(binvect) == len(self.pas)
        assert all(x == 0 or x == 1 for x in binvect)
        for k, v in enumerate(binvect):
            if v == 1:
                self.labels[k] = label

    def _is_at_beginning(self, binvect):
        """
        Returns True if some labels were changed
        (binvect must be an iterable of 0 and 1)
        """

        first = min(k for (k, v) in enumerate(binvect) if v == 1)
        steps_between = np.arange(first)
        if 0 <= len(steps_between) <= self._n_to_end:
            self._set_label(binvect, still)
            for k in steps_between:
                self.labels[k] = still
            return True
        else:
            return False

    def _is_at_end(self, binvect):
        """
        Returns True if some labels were changed
        (binvect must be an iterable of 0 and 1)
        """
        last = max(k for (k, v) in enumerate(binvect) if v == 1)
        n_step = len(self.pas)
        steps_between = np.arange(last + 1, n_step)
        if 0 <= len(steps_between) <= self._n_to_end:
            self._set_label(binvect, still)
            for k in steps_between:
                self.labels[k] = still
            return True
        else:
            return False

    def _close_to_uturn(self, binvect):
        """
        Returns True if some labels were changed
        (binvect must be an iterable of 0 and 1)
        """
        if uturn not in self.labels:
            self._set_label(binvect, uturn)
            return True
        else:
            first = min(k for (k, (v, l))
                        in enumerate(zip(binvect, self.labels))
                        if v == 1 or l == uturn)
            last = max(k for (k, (v, l))
                       in enumerate(zip(binvect, self.labels))
                       if v == 1 or l == uturn)

            steps_between = [k for (k, (v, l))
                             in enumerate(zip(binvect, self.labels))
                             if first <= k <= last
                             and v == 0
                             and l == walk]
            # peu de pas entre l'ancien demi-tour et le nouveau
            if 0 <= len(steps_between) <= self._n_to_uturn:
                self._set_label(binvect, uturn)
                for k in steps_between:
                    self.labels[k] = uturn
                return True
            else:
                return False

    def segmentation(self,level=None, toplevel=None):
        """
        Ici on fait la segmentation. La fonction affecte l'attribut
        self.labels.
        Si les pas détectés sont au début ou à la fin, alors ils sont classés
        dans la catégorie "still".
        Si aucun demi-tour n'était déjà détecté, ces pas deviennent le
        demi-tour.
        Si les pas sont proches d'un demi-tour déjà détecté, ils sont fusionnés
        s'ils sont proches. Sinon un crée un warning.

        """


        if level is not None:
            self.level = level
        if toplevel is not None:
            self.toplevel = toplevel

        above = self._is_above(consider=[walk])

        # si rien n'a été détecté, il se ne passe rien
        if all(x == 0 for x in above):
            return
        # Si les pas sont au début, on les range dans "still"
        if self._is_at_beginning(binvect=above):
             return
        # # Si les pas sont à la fin, on les range dans "still"
        elif self._is_at_end(binvect=above):
             return
        # S'il n'y a pas de demi-tour, les pas détectés sont étiquetés comme
        # appartenant au demi-tour.
        # Sinon, les pas détectés sont proche du demi-tour, on les fusionne
        elif self._close_to_uturn(binvect=above):
            return
        # Ici, un demi-tour potentiel a été detecté mais il ne correspond à
        else:
            # aucune situation prévue, donc on crée un warning.
            warnings.warn(
                "Potential u-turn during the walk: " + str(self.e.fname),
                exc.TooManyUTurns)
            self._set_label(above, uturn)
            return

    def seg_from_labels(self):
        """
        Returns the segmentation times:
        [walk start, uturn start, uturn end, walk end]
        """
        if uturn not in self.labels:
            self.seg = [0, 0, 0, 0]
            return self.seg

        res = list()
        # first step of the walk
        step = min(k for (k, lab) in enumerate(self.labels)
                   if lab == walk)
        res += [self.pas[step][0]]
        # first step of the u-turn
        step = min(k for (k, lab) in enumerate(self.labels)
                   if lab == uturn)
        res += [self.pas[step][0]]
        # last step of the u-turn
        step = max(k for (k, lab) in enumerate(self.labels)
                   if lab == uturn)
        res += [self.pas[step][1]]
        # last step of the walk
        step = max(k for (k, lab) in enumerate(self.labels)
                   if lab == walk)
        res += [self.pas[step][1]]

        self.seg = res
        #plot(self)
        return self.seg

    #def compute(self, level=None):
    def compute(self,e):
        self.labels = [walk] * len(self.pas)  # initialisation des labels

        count = 0
        m = max(f for (f, l) in zip(self.feats, self.labels) if l == walk)
        conditions = True
        ut=0
        while conditions:
            # on fait la segmentation
            # tant qu'il n'y a pas de demi-tour détecté on relaxe la condition
            # toplevel

            if uturn not in self.labels:
                tmp_toplevel = self.toplevel
                self.segmentation(level=self.level, toplevel=self.level)
                self.toplevel = tmp_toplevel
            else:
                self.segmentation(level=None, toplevel=None)
                # ut+=1
                # if ut==1:
                #     break

            # on met les conditions à jour
            count += 1
            m = max(f for (f, l) in zip(self.feats, self.labels) if l == walk)
            conditions = ((count < self._maxiter) or
                          (uturn not in self.labels) or
                          (m > self.toplevel))

        # warnings sur la qualité de la segmentation.
        n_uturn = len([1 for (l1, l2) in zip(self.labels[:-1], self.labels[1:])
                       if l1 != uturn and l2 == uturn])
        if uturn not in self.labels:
            warnings.warn("No u-turn found: " + self.e.fname, exc.NoUTurn)
            print(self.seg_from_labels())
        elif n_uturn > 1:
            warnings.warn("This exercise has too many uturns: " +
                          self.e.fname, exc.TooManyUTurns)

        return self.seg_from_labels()


def plot(s):
    import matplotlib.pyplot as plt
    import pylab
    fig, ax = plt.subplots(figsize=(20, 10))
    legend = list(zip([uturn, walk, still],
                      ["red", "blue", "green"]))
    temps = [np.mean([start, end]) for (start, end) in s.pas]
    signal = s.e.get_signal("rCRX").T.flatten()
    signal -= min(signal)
    signal /= max(signal)
    print(signal)
    plt.plot(1-signal)
    for lab, leg in legend:
        points = [(t, f)
                  for (t, f, l) in zip(temps, s.feats, s.labels) if l == lab]
        if len(points) > 0:
            x, y = zip(*points)
            ax.scatter(x, y, marker="o", s=50, color=leg)
    ax.vlines(s.seg, ymin=0, ymax=1, linewidth=2)
    ax.hlines(s.level, xmin=0, xmax=s.e.data_earth.shape[1], linewidth=2)
    ax.hlines(s.toplevel, xmin=0, xmax=s.e.data_earth.shape[1], linewidth=2, linestyle="dashed")
    ax.set_title(s.e.fname)
    pylab.show()
    return fig, ax