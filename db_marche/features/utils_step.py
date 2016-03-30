__author__ = 'charles'
import logging
import scipy
import numpy as np
def extract_steps_rDAZ(exo):
    '''Returns the list of signals rDAZ for the annotated steps'''
    #steps = exo.steps_annotation[0]
    steps = exo.steps[0]
    z = exo.rDAZ.T
    #z= z - z.mean()
    return [z[start:end] for (start, end) in steps]


def extract_steps_rGAZ(exo):
    '''Returns the list of signals rGAZ for the annotated steps'''
    #steps = exo.steps_annotation[1]
    steps = exo.steps[1]
    z = exo.rGAZ.T
    #z= z - z.mean()
    return [z[start:end] for (start, end) in steps]


def extract_hs_from_step(signal):
    ''' First local maximum after the signal goes negative to positive.'''
    # premier passage des négatifs aux positifs
    index = next(i for i, v in enumerate(signal[1:], start=1)
                 if v > 0 and signal[i - 1] < 0)
    # premier maximum local après le passage au dessus de 0
    index = next(i for i, v in enumerate(signal[index:-2], start=index)
                 if v > signal[i - 1] and v > signal[i + 1])
    return index


def extract_steps_rDRY(exo):
    """
    Returns the list of signals rDRY for the annotated steps
    :param exo: class Exercice
    :return: list of univariate numpy arrays
    """
    #steps = exo.steps_annotation[0]
    steps = exo.steps[0]

    z = exo.rDRY.T
    z= z - z.mean()
    return [z[start:end] for (start, end) in steps]


def extract_steps_rGRY(exo):
    """
    Returns the list of signals rGRY for the annotated steps
    :param exo: class Exercice
    :return: list of univariate numpy arrays
    """
    #steps = exo.steps_annotation[1]
    steps = exo.steps[1]
    z = exo.rGRY.T
    z= z - z.mean()
    return [z[start:end] for (start, end) in steps]


def extract_to_from_step(signal):
    return 5

def extract_right_to_from_exo(exo):
    """
    Extract the list of the right toe off times from an exercise.
    :param exo: class Exercice
    :return: list of indexes.
    """
    step_sig = extract_steps_rDRY(exo)
    res = list()
    #for z, (start, end) in zip(step_sig, exo.steps_annotation[0]):
    for z, (start, end) in zip(step_sig, exo.steps[0]):
        r=peaks(-z.flatten())
        try:
            #res.append(extract_to_from_step(z) + start)
            res.append(r[0]+ start)
        except:
            # In case there is no toe off detected we replace by the beginning
            # of the annotated step
            #logging.warning("No toe off detected in some steps: " + exo.fname)
            res.append(start)
    return res


def extract_left_to_from_exo(exo):
    """
    Extract the list of the left toe off times from an exercise.
    :param exo: class Exercice
    :return: list of indexes.
    """
    step_sig = extract_steps_rGRY(exo)
    res = list()
    #for z, (start, end) in zip(step_sig, exo.steps_annotation[1]):
    for z, (start, end) in zip(step_sig, exo.steps[1]):
        r=peaks(-z.flatten())
        try:
            #res.append(extract_to_from_step(z) + start)

            res.append(r[0]+ start)
        except:
            # In case there is no toe off detected we replace by the beginning
            # of the annotated step
            #logging.warning("No toe off detected in some steps: " + exo.fname)
            res.append(start)
    return res


def extract_right_hs_from_exo(exo):
    """
    Extract the list of the right heel strike times from an exercise.
    :param exo: class Exercice
    :return: list of indexes.
    """
    step_sig = extract_steps_rDAZ(exo)
    res = list()
    #for z, (start, end) in zip(step_sig, exo.steps_annotation[0]):
    for z, (start, end) in zip(step_sig, exo.steps[0]):
        r=peaks(-z.flatten())
        try:
            #res.append(extract_hs_from_step(z) + start)
            res.append(r[1]+ start)
        except:
            # In case there is no toe off detected we replace by the beginning
            # of the annotated step
            #logging.warning("No heel strike detected in some steps: "+ exo.fname)
            res.append(end)
    return res


def extract_left_hs_from_exo(exo):
    """
    Extract the list of the left heel strike times from an exercise.
    :param exo: class Exercice
    :return: list of indexes.
    """
    step_sig = extract_steps_rGAZ(exo)
    res = list()
    #for z, (start, end) in zip(step_sig, exo.steps_annotation[1]):
    for z, (start, end) in zip(step_sig, exo.steps[1]):
        r=peaks(-z.flatten())
        #print("Left HS:",z)
        try:
            #res.append(extract_hs_from_step(z) + start)
            res.append(r[1]+ start)
        except:
            # In case there is no toe off detected we replace by the beginning
            # of the annotated step
            #logging.warning("No heel strike detected in some steps: "+ exo.fname)
            res.append(end)
    return res
    """
    First local maximum after the signal goes negative to positive.
    :param signal: signal of a step
    :return: int
    """
    # premier passage des négatifs aux positifs
    index = next(i for i, v in enumerate(signal[1:], start=1)
                 if v > 0 and signal[i - 1] < 0)

def peaks(vector):

    from peakutils.peak import indexes
    vector=cleansignal(vector)
    thres=1/max(abs(vector))
    indexes2 = indexes(vector, thres, min_dist=len(vector)/2)
    if len(indexes2)==2:

        s=vector[indexes2[0]:indexes2[1]]
        #print(s)
        zero_crossings = np.where(np.diff(np.sign(s)))[0]
        if len(zero_crossings)==2:
            indexes2[1]=zero_crossings[1]+indexes2[0]
    return indexes2

def cleansignal(s):
    w = scipy.fftpack.rfft(s)
    spectrum = w**2
    cutoff_idx = spectrum < (spectrum.max()/1000)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    return scipy.fftpack.irfft(w2)