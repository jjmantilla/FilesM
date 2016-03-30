from setuptools import setup, find_packages,Extension
from Cython.Build import cythonize
import numpy
setup(name='db_marche',version='0.1',packages=find_packages(),extensions=[numpy.get_include()])
