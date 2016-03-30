import numpy as np
from collections import defaultdict

from db_marche import Database, DATA_DIR
from db_marche.process.pattern import Pattern

N_TEMPLATES = 5


def mk_pattern_library(db=None, n_templates=N_TEMPLATES,
                       save=False, debug=-1):
    '''Create the pattern library used to detect the steps
    and store it to the right place

    Parameters
    ----------
    n_templates
    '''
    if db is None:
        db = Database(debug=debug)

    # Select all the different atteinte with more than n_template patient
    att = defaultdict(lambda: set())
    for m in db.meta:
        s = att[m['atteinte']]
        s.add(m['code'])
        att[m['atteinte']] = s
    atteintes = []
    for k, v in att.items():
        if len(v) > n_templates:
            atteintes += [k]

    # Load one exercise per patient and per
    l_ex = []
    for a in atteintes:
        l_ex += db.get_data(limit=n_templates, atteinte=a, code='max1',
                            load_args=dict(load_steps=False,
                                           load_seg=False))

    patterns = []
    for ex in l_ex:
        # Chose a random foot and a random step
        ok = False
        while not ok:
            foot = np.random.rand() >= .5
            n_step = np.random.randint(len(ex.steps_annotation[foot]))
            step = ex.steps_annotation[foot][n_step]
            ok = step[1] - step[0] > 1
            sig = ex.data_earth[6*foot+2]
            ok &= np.sum(sig*sig) > 1e-7

        # Determine step position (segment, n_step_segment, pos_segment)
        position = (step[0]+step[1])/2
        segment = 'forward'
        b_segment, e_segment = ex.seg_annotation[:2]
        if position >= ex.seg_annotation[1]:
            segment = 'uturn'
            b_segment, e_segment = ex.seg_annotation[1:3]
        if position >= ex.seg_annotation[2]:
            segment = 'backward'
            b_segment, e_segment = ex.seg_annotation[2:]
        nb_step_segment = 0
        for f in range(2):
            for ns, s in enumerate(ex.steps_annotation[f]):
                pos = (s[0]+s[1])/2
                if b_segment <= pos <= e_segment:
                    nb_step_segment += 1
                if (f, ns) == (foot, n_step):
                    pos_segment = nb_step_segment

        # Starting foot
        SR1 = ex.steps_annotation[0][0][0]
        SR2 = ex.steps_annotation[1][0][0]
        takeoff_foot = int(SR2 >= SR1)
        is_takeoff = takeoff_foot == foot

        meta_step = dict(foot='left' if foot else 'right',
                         segment=segment, pos_segment=pos_segment,
                         nb_step_segment=nb_step_segment,
                         id_step=n_step, is_takeoff=is_takeoff)
        meta_step.update(ex.meta)
        patterns += [Pattern(dict(meta_step, coord='AV'),
                             ex.data_earth[6*foot+2, step[0]:step[1]]),
                     Pattern(dict(meta_step, coord='AZ'),
                             ex.data_sensor[6*foot+2, step[0]:step[1]]),
                     Pattern(dict(meta_step, coord='RY'),
                             ex.data_sensor[6*foot+4, step[0]:step[1]])]

    if save:
        import os.path as osp
        db_pattern = osp.join(DATA_DIR, 'Steps', 'DB_steps.npy')
        np.save(db_pattern, patterns)

    return patterns


def mk_rand_pattern_library(P=25, db=None,
                            debug=-1, **kwargs):
    '''Create the pattern library used to detect the steps
    and store it to the right place

    Parameters
    ----------
    n_templates
    '''
    if db is None:
        db = Database(debug=debug)
    l_ex = db.get_data(limit=P, code='max1', **kwargs)

    patterns = []
    codes = []
    for ex in l_ex:
        # Chose a random foot and a random step
        ok = False
        while not ok:
            foot = np.random.rand() >= .5
            n_step = np.random.randint(len(ex.steps_annotation[foot]))
            step = ex.steps_annotation[foot][n_step]
            ok = step[1] - step[0] > 10
            sig = ex.data_earth[6*foot+2]
            ok &= np.sum(sig*sig) > 1e-7
        codes += [ex.code]

        # Determine step position (segment, n_step_segment, pos_segment)
        position = (step[0]+step[1])/2
        segment = 'forward'
        b_segment, e_segment = ex.seg_annotation[:2]
        if position >= ex.seg_annotation[1]:
            segment = 'uturn'
            b_segment, e_segment = ex.seg_annotation[1:3]
        if position >= ex.seg_annotation[2]:
            segment = 'backward'
            b_segment, e_segment = ex.seg_annotation[2:]
        nb_step_segment = 0
        for f in range(2):
            for ns, s in enumerate(ex.steps_annotation[f]):
                pos = (s[0]+s[1])/2
                if b_segment <= pos <= e_segment:
                    nb_step_segment += 1
                if (f, ns) == (foot, n_step):
                    pos_segment = nb_step_segment

        # Starting foot
        SR1 = ex.steps_annotation[0][0][0]
        SR2 = ex.steps_annotation[1][0][0]
        takeoff_foot = int(SR2 >= SR1)
        is_takeoff = takeoff_foot == foot

        meta_step = dict(foot='left' if foot else 'right',
                         segment=segment, pos_segment=pos_segment,
                         nb_step_segment=nb_step_segment,
                         id_step=n_step, is_takeoff=is_takeoff)
        meta_step.update(ex.meta)
        patterns += [Pattern(dict(meta_step, coord='AV'),
                             ex.data_earth[6*foot+2, step[0]:step[1]]),
                     Pattern(dict(meta_step, coord='AZ'),
                             ex.data_sensor[6*foot+2, step[0]:step[1]]),
                     Pattern(dict(meta_step, coord='RY'),
                             ex.data_sensor[6*foot+4, step[0]:step[1]])]

    return patterns, codes
