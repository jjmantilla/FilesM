import pickle
import logging
import numpy as np
import os.path as osp

from . import DATA_DIR, DESC
from .utils import sane_str

logger = logging.getLogger("Database")


class Exercise(object):
    """Class containing one exercice

    Description
    -----------
    This class handles loading and accessing acquisitions from TechnoConcept
    and X-sens captors.
    It permits to access

    Usage
    -----
    Construct an instance of this class with a fname format as:
        PatientCode-Eye-Enum-Sensor-PARTX.csv

    The segmentation, the step detection and the feature computation
    are done automatically on the first load of the exercices and are
    then stored in pickle files in the DATA_DIR folder.

    To access one part of the signal, you can use the method get_signal
    taking a descriptor of the part of the signal you need

    Parameters
    ----------
    meta: dict
        All information loaded about this exercise
    recompute: str, optional (default: '')
        Recompute the segmentation, the steps or the features
        Possible value are ['step', 'seg', 'feat', 'all']
        A join can be made with '|'
    """
    def __init__(self, meta, recompute=''):
        self.meta = meta
        self.fname = meta['fname']

        pkl_file = self._get_pkl_name()
        if osp.exists(pkl_file):
            # we load the exercise from a pickle file if it exists
            self._load()
        else:
            recompute = 'all'
        self.load_signal()

        #print("Recompute : ", recompute)


        if not hasattr(self, 'g_earth'):
             self._compute_g()
        recompute = recompute.replace('all', 'sig|step|seg|feat')
        # Compute the parts of the exercice
        for part, load_part in [
                                #('sig', self.load_signal),
                                ('step', self.load_steps),
                                ('seg', self.load_segmentation),
                                ('feat', self.load_feat)
        ]:
            if part in recompute:
                load_part()

        # Compute the gravity


        # Create place holder for manual annotation
        if not hasattr(self, 'seg_annotation'):
            self.seg_annotation = None
        if not hasattr(self, 'steps_annotation'):
            self.steps_annotation = None

    def load_signal(self):
        from .process.signal_loader import SignalLoader
        loader = SignalLoader(self.meta)
        self.data_sensor = loader.data_sensor
        self.data_earth = loader.data_earth

    def load_manual_segmentation(self, reloading=False):
        fname = osp.basename(self.fname)
        labm_seg = osp.join(DATA_DIR, 'Manual_Annotation',
                            fname.replace('PARTX.csv', 'seg.labm'))
        labm_seg = labm_seg.replace('-YO', '')
        labm_seg = labm_seg.replace('-Xsens', '').replace('-TCon', '')
        if osp.exists(labm_seg):
            seg_annotation = []
            with open(labm_seg, 'r') as f:
                for line in f.readlines()[:4]:
                    seg_annotation += [int(line)]
            self.seg_annotation = seg_annotation
        else:
            logger.info('No manual segmentation for : {}'.format(self.fname))
            self.seg_annotation = None
        return

    def load_manual_steps(self, reloading=False):
        fname = osp.basename(self.fname)
        labm_seg = osp.join(DATA_DIR, 'Steps',
                            fname.replace('PARTX.csv', 'step.labm'))
        if osp.exists(labm_seg):
            try:
                steps_annotation = [[]]
                with open(labm_seg, 'r') as f:
                    for line in f.readlines():
                        if line[0] == '/':
                            steps_annotation.append([])
                        else:
                            steps_annotation[-1] += [[
                                int(v) for v in line.split(';')]]
                if len(steps_annotation[1]) > 0:
                    self.steps_annotation = steps_annotation
            except ValueError:
                logger.info('Manual annotation fail   : {}'
                            ''.format(self.fname))
                self.steps_annotation = None
        else:
            logger.info('No manual annotation for : {}'.format(self.fname))
            self.steps_annotation = None

    def load_segmentation(self, recompute=False):
        '''
        Method to perform a phase segmentation for the exercice
        '''
        logger.debug('Compute segmentation')
        from .process.Segmentation import Segmentation as seg
        Segmentation = seg(self)
        Segmentation.segmentation(level=0.2,toplevel=0.6)
        self.seg_annotation = Segmentation.compute(self)

    def load_steps(self, interactive=False, recompute=False):
        '''
        Method to perform a step detection for the exercice
        '''
        from .process.step_detection import StepDetection
        logger.debug('Compute steps')
        stepDetection = StepDetection()
        self.steps, self.steps_meta = stepDetection.compute_steps(self)

    def load_feat(self, recompute=False):
        '''
        Method to perform the computation of features for the exercice
        '''
        logger.debug('Compute features')
        from db_marche._import_features import compute_features
        self.feats, self.feats_desc = compute_features(self)


    def _compute_g(self):
        '''Compute the gravity from the first seconds
        of the recording using the mod of the acceleration
        '''
        try:
            N = 400
            g_earth, g_sensor = [], []
            split = np.round(np.linspace(0, N, round(np.sqrt(N))))
            for j0, j1 in zip(split[:-1], split[1:]):
                g_sensor += [np.median(self.data_sensor[:, j0:j1], axis=1)]
                g_earth += [np.median(self.data_earth[:, j0:j1], axis=1)]

            g_earth = np.mean(g_earth, axis=0)
            g_sensor = np.mean(g_sensor, axis=0)

            for i in range(3):
                g_sensor[3+i::6] = 0
                g_earth[3+i::6] = 0
            self.g_earth = g_earth.reshape((-1, 1))
            self.g_sensor = g_sensor.reshape((-1, 1))
        except TypeError:
            self.g_earth = None
            self.g_sensor = None
            logger.warning("Failed to compute gravity for exercise {}"
                           "".format(self.id))

    def get_signal(self, desc='DAZS'):
        '''Return one signal matching the descriptor

        Parameters
        ----------
        desc: str, optional (default: DAZS)
            Select a particular signal with 3 characters
                - D(roit) / G(auche) / T(ete) / C(einture)
                - A(cceleration)/R(otation)
                - X / Y / Z / H
            You can select segments of the signal by adding
                - S: whole Signal
                - F: Forward part
                - B: Backward part
                - U: U-turn part
        '''
        k = np.zeros(1, np.int)
        c = 0
        sig = self.data_earth
        g = self.g_earth
        if desc[0] == 'r':
            sig = self.data_sensor
            g = self.g_sensor
            c += 1
        sig = sig-g

        for opt, s in [('DGCT', 6), ('AR', 3), ('XYZ', 1)]:
            if c < len(desc) and desc[c] in opt:
                k += DESC.get(desc[c], 0)
                c += 1
            else:
                k = np.r_[[vp+s*v for vp in k for v in range(len(opt))]]
        if desc[c:] == '' or desc[c] == 'S':
            return sig[k]

        if not np.all([l in 'FBU' for l in desc[c:]]):
            print('Bad signal selection')

        sel = []
        for l in desc[c:]:
            if l == 'F':
                sel += list(range(self.seg_annotation[0],
                                  self.seg_annotation[1]))
            elif l == 'U':
                sel += list(range(self.seg_annotation[1],
                                  self.seg_annotation[2]))
            elif l == 'B':
                sel += list(range(self.seg_annotation[2],
                                  self.seg_annotation[3]))
        return sig[k][:, sel]

    def __getattr__(self, k):
        try:
            return super(Exercise, self).__getattribute__(k)
        except AttributeError:
            if k == 'meta':
                raise AttributeError('Try to get meta before setting it')
            try:
                return self.meta[sane_str(k)]
            except (KeyError, AssertionError):
                if len(k) < 5 and np.all([c in 'rDGCTARXYZSFUB' for c in k]):
                    return self.get_signal(k)
                else:
                    raise AttributeError("{}({}) no attribute {}"
                                         "".format(self.__class__.__name__,
                                                   self.fname, k))

    def __setattr__(self, key, value):
        """
        We overload the setattr method to intercept any change in
        the Exercise instance.
        """
        super().__setattr__(key, value)
        if key not in ('meta', 'fname'):
            # logger.debug('We dump the Exercise instance in a pickle'
            #              ' file because it has changed.')
            self._dump()

    def _dump(self):
        pkl_file = self._get_pkl_name()
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)
            f.close()

    def _load(self):
        pkl_file = self._get_pkl_name()
        with open(pkl_file, 'rb') as f:
            tmp_dict = pickle.load(f)
            f.close()
        del tmp_dict['meta']
        self.__dict__.update(tmp_dict)

    def _get_pkl_name(self):
        file_code = self.fname.replace('-Xsens-PARTX.csv', '')
        file_code = file_code.replace('-TCon-PARTX.csv', '').replace('-YO', '')
        pkl_file = osp.join(DATA_DIR, 'Exo', file_code) + '.pkl'
        return pkl_file



