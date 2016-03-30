import pyximport
pyximport.install()

#from .aux_step_detection import *
from ..cython_step.aux_step_detection import *
