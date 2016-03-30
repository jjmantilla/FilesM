
from setuptools import setup, find_packages
setup(name='db_marche',
      version='0.1',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'matplotlib',
                        'pymongo'
                        
      ],
      )


