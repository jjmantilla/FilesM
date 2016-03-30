import logging
import sys
import numpy as np
from collections import defaultdict

from .utils import sane_str

out = sys.stdout
logger = logging.getLogger("Database")
logger.setLevel(logging.DEBUG)


class _Database(object):
    """Class to handle the patient database and do lazy loading

    Usage
    -----
    get_exercices: method to get a list of exercices matching a certain query
        and a certain format.

    """

    def __init__(self, settings, debug=1):
        '''Constructor of the Database object

        Parameters
        ----------
        debug: int, optional (default: 1)
            debug level {-1: None, 0: Warning, 1: Info, 2: Debug}
        settings: str
            path to the setting file in json that is loaded for the
            Database instance
        '''
        import json
        settings = __file__.replace('_database.py', settings)
        with open(settings) as f:
            self.data_settings = json.load(f)
        self.data_folder = self.data_settings['data_folder']
        if self.data_folder[0] != '/':
            from os.path import dirname, join
            dname = dirname(__file__)
            self.data_folder = join(dname, self.data_folder)
        if debug > -1:
            self._init_log(debug)

        self.debug = debug
        self.meta = []
        self.load_data_info()
        self.N = len(self.meta)
        self.objects = [None]*self.N

    def _init_log(self, debug):
        '''Init logging interface
        '''
        level = (3 - debug)*10
        if len(logger.handlers) == 0:
            ch = logging.StreamHandler()
            ch.setLevel(level)

            formatter = logging.Formatter('%(name)s - '
                                          '%(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            logger.debug("Init logger")

    def load_data_info(self):
        '''Import the meta data for all objects in the DB
        It permits to initiate the query mechanism by filling
        the meta list
        '''
        raise NotImplementedError("Database does not implement load_data_info")

    def load_data_object(self, index):
        '''Import the data object with id
        '''
        raise NotImplementedError("Database does not implement "
                                  "load_data_object")

    def get_data(self, limit=None, seed=None, weigths=None, load_args={},
                 **query):
        '''Load data from the base matching the given query
        limit: int, optional (default: None)
            Number maximal of exercices loaded
        seed: uint, optional (default: None)
            Used to see RNG. must be lower than 2147483648
        weigths: list, optional (default: None)
            Weighting options. List of pair (w, q)
            with w the weight of the class, and q the
            class specific selection query
        **query: querry definition
        '''
        self._reset(seed)
        data_list = []
        if weigths is not None:
            for w, mod in weigths:
                q = dict(query)
                q.update(mod)
                data_list += self.get_data(
                    limit=limit*w, seed=np.random.randint(0, 2147483648),
                    **q)
            return data_list
        if limit is not None:
            limit = int(limit)

        queries = lambda meta: np.all(
            [self._query(k, v, meta[sane_str(k)])
             for k, v in query.items()])
        if self.debug > 0:
            out.write('Loading Data: {:7.2%}'.format(0))
            out.flush()
        for i, m in enumerate(self.meta):
            if queries(m):
                if self.objects[i] is None:
                    self.objects[i] = self.load_data_object(i, **load_args)
                do = self.objects[i]
                if do is not None:
                    data_list += [do]
                    for k in list(self._query_max_key):
                        self._query_data[m[k]] += 1
            if limit is not None:
                if self.debug > 0:
                    out.write('\rLoading Data: {:7.2%}'
                              ''.format(len(data_list)/limit))
                    out.flush()
                if len(data_list) >= limit:
                    break
            elif self.debug > 0:
                out.write('\rLoading Data: {:7.2%}'.format(i/len(self.meta)))
                out.flush()
        if self.debug > 0:
            print('\rLoading Data: {:7}'.format('Done'))
        return data_list

    def _query(self, key, pattern, value):
        '''Permits to select data objects which match
        the given query.
        '''
        if pattern is None or pattern == '':
            return True
        if callable(pattern):
            return pattern(value)
        if type(pattern) is list:
            return np.all([self._query(key, p, value) for p in pattern])
        if type(pattern) is int:
            return value == pattern
        if 'max' in pattern:
            val = int(pattern.replace('max', ''))
            self._query_max_key.add(key)
            return self._query_data[value] < val
        if '<' == pattern[0]:
            sup_val = float(pattern.replace('<', ''))
            return float(value) < sup_val
        if '>' == pattern[0]:
            inf_val = float(pattern.replace('>', ''))
            return float(value) > inf_val
        if pattern[0] != '^':
            return np.any([(p in value) for p in pattern.split('|')])
        else:
            return np.all([(p not in value)
                           for p in pattern[1:].split('|')])

    def _reset(self, seed):
        assert self.N > 0, "No data available in this Database"

        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(np.random.randint(0, 2147483648))
        i0 = np.argsort([m['code']+str(m['id_exercise'])
                        for m in self.meta])
        order = np.arange(self.N)
        np.random.shuffle(order)
        self.meta = [self.meta[i0[i]] for i in order]
        self.objects = [self.objects[i0[i]] for i in order]

        self._query_data = defaultdict(lambda: 0)
        self._query_max_key = set()
