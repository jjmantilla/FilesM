import numpy as np


class Pattern:
    '''Structure to contain the patterns information

    Attributes:
    -----------
    meta:
    vect: signal of the pattern
    coord: axis of the pattern
    '''
    def __init__(self, meta, data):
        self.meta = meta
        self.data = data - np.mean(data)
        self.coord = meta['coord']
        self.l_pattern = len(data)
        self.n_pattern = np.sqrt(np.sum(data*data))
