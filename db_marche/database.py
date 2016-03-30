import logging
import os

from . import mk_db_structure
from .exercise import Exercise
from ._database import _Database

from .utils import sane_update

logger = logging.getLogger("Database")


class Database(_Database):
    """Class to handle the patient database and do lazy loading

    Usage
    -----
    get_exercices: method to get a list of exercices matching a certain query
        and a certain format.

    """
    def __init__(self, settings=None, debug=1):
        '''Constructor of the Database object

        Parameters
        ----------
        debug: int, optional (default: 1)
            debug level {-1: None, 0: Warning, 1: Info, 2: Debug}
        '''
        if settings is None:
            settings = 'marche_setting.json'
        super(Database, self).__init__(settings, debug=debug)
        self.n_samples = len(self.meta)
        logger.debug("Init Database")

    def load_data_info(self):
        '''Import the database of all the patients in the DATA_DIR
        directory, keep there name and create Patient objects.
        '''
        import glob
        from os.path import join as j

        DATA_DIR = self.data_folder
        mk_db_structure(DATA_DIR)

        # Select code_name of all the patient in the database
        meta = []
        done = set()
        for pattern in [j(DATA_DIR, 'Data', '*Tete.csv'),
                        j(DATA_DIR, 'Raw', '*672.txt'),
                        j(DATA_DIR, 'Raw', '*datas-1.txt')]:
            for fname in glob.glob(pattern):
                fname = fname.split(os.sep)[-1]
                info = self._process_name(fname)
                if info is not None and info['fname'] not in done:
                    meta += [info]
                    done.add(info['fname'])
        self.meta = meta

    def load_data_object(self, index, **kwargs):
        meta = self.meta[index]
        try:
            return Exercise(meta=meta, **kwargs)
        except AssertionError as e:
            logger.warning(
                'Did not load data for {}_{}\n\t{}'
                ''.format(meta['code'], meta['id_exercice'], e)
            )

    def _process_name(self, fname):
        import json
        import re
        try:
            g = re.search(r'^(\w{3}-\w{3,4})-YO-([0-9]+)-(Xsens|TCon|000)',
                          fname).groups()
            code = g[0]
            id_exercise = g[1]
            sensor = g[2] if g[2] != '000' else 'Xsens'
            name = fname.replace('Tete', 'PARTX')
            name = name.replace('000_00340672.txt', 'Xsens-PARTX.csv')
            name = name.replace('datas-1.txt', 'PARTX.csv')
            meta_exercice = dict(code=code, id_exercise=int(id_exercise),
                                 id=code+id_exercise,
                                 fname=name, sensor=sensor)
            with open(os.path.join(self.data_folder, 'Labels',
                                   code+'-'+id_exercise+'-label.js')) as f:
                lab = json.load(f)
                sane_update(meta_exercice, lab)
            return meta_exercice
        except AttributeError:
            logger.debug('No code in {}'.format(fname))
        except IOError:
            logger.warning('No labels for {} - {}'
                            ''.format(code, id_exercise))
        return None
