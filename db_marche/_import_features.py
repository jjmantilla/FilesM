import os
import glob
from os.path import join as j
from importlib import import_module


def compute_features(ex, package=['db_marche.features']):
    cc = 0
    feats = {}
    pname = os.path.abspath(os.path.join(os.path.dirname(__file__), 'features'))

    # List the directories present in the given package

    # For each directory with name not starting with _
    # We load all the files and add it to the features
    # dictionnary

    l_dirs = glob.glob(j(pname, '*',''))

    for ldir in l_dirs:
        dirname = ldir.split(os.sep)[-2]

        if dirname[0] != '_':

            n, s_feats = compute_features(ex, package+[dirname])
            if n > 0:
                feats.update(s_feats)
                cc += n

    # List the module present in the given package
    # For each module, compute all the features and
    # add them to the features dictionary
    for m in glob.glob(j(pname, '*.py')):
        fname = m.split(os.sep)[-1][:-3]

        if fname[0] != '_':
            module = '.'.join(package + [fname])
            s_feats = load_feature(ex, module=module)
            feats.update(s_feats)
            cc += len(s_feats)
    return cc, feats


def load_feature(ex, module):
    '''Compute for ex all the features of the present module

    Return a dictionary of features
    '''
    feats = {}
    mm = import_module(module)
    for s in mm.__dir__():
        if s.startswith('feat'):
            fun_feat = mm.__getattribute__(s)
            try:
                feats.update(fun_feat(ex))
            except:
                import traceback
                msg = traceback.format_exc()
                print('Feature {} failed  on exercise {}-{} with error:\n{}'
                      ''.format(s, ex.code, ex.id_exercice, msg))
    return feats
