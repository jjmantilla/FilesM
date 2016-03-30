import numpy as np


def count_hit(seg_label, seg):
    '''Count the #segment's median in seg that can be associated
    to a segment in seg_label (without replacement)

    Parameters
    ----------
    seg_label: list(2-uples)
        segmentation labels
    seg: list(2-uples)
        segmentation to score

    Return: #TruePositif for counting metric
    '''
    count = 0
    used = np.zeros(len(seg_label))
    for tstart, tend in seg:
        pos = (tend+tstart)/2
        for i, (s0, s1) in enumerate(seg_label):
            if s0 < pos < s1 and not used[i]:
                used[i] = 1
                count += 1
                break
    return count


def score_echantillon(seg_label, seg):
    '''Compare the 2 segmentations in each sample step

    Parameters:
    seg_label: list(2-uples)
        segmentation labels
    seg: list(2-uples)
        segmentation to score

    Return: list res
        res[0] -> Recall - #TruePositif / #LabelPositif
        res[1] -> Precision - #TruePositif / #PredPositif
    '''
    N = max(seg_label.max(), seg.max())
    sy = np.zeros(N)
    for (d, t) in seg_label:
        sy[d:t] = 1
    sp = np.zeros(N)
    for (d, t) in seg:
        sp[d:t+1] = 1
    R = (sy*sp).sum()/sy.sum()
    P = (sy*sp).sum()/sp.sum()
    return [R, P]


def hamming(seg_label, seg):
    return (hamingd(seg_label, seg) + hamingd(seg, seg_label)) / 2


def hamingd(seg1, seg2):
    d_ham = 0
    L = 0
    for tstart, tend in seg1:
        L += tend-tstart
        max_junc = 0
        for s0, s1 in seg2:
            junc = max(0, min(s1, tend)-max(s0, tstart))
            max_junc = max(max_junc, junc)
        assert(max_junc <= tend-tstart)
        d_ham += tend-tstart - max_junc
    return d_ham/(L+(L == 0))


def count(seg_label, seg):
    count = 0
    used = np.zeros(len(seg))
    L = len(seg_label)
    for tstart, tend in seg_label:
        for i, (s0, s1) in enumerate(seg):
            junc = max(0, min(s1, tend)-max(s0, tstart))/(tend-tstart)
            junc *= ((s1-s0) < 1.3*(tend-tstart))
            if junc > 0.6 and not used[i]:
                used[i] = 1
                count += 1
                break
    if len(seg_label) == 0:
        print('Issue!!!')
    return count/(L+(L == 0))

