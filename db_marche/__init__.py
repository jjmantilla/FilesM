from os import path, mkdir
# Define const variables for the module

# Folder containing the data files
import json
settings = __file__.replace('__init__.py', 'marche_setting.json')
with open(settings) as f:
    data_settings = json.load(f)
DATA_DIR = data_settings['data_folder']


def mk_db_structure(folder):
    STORAGE_DIRS = ['Exo', 'Labels', 'Raw', 'Manual_Annotation', 'Import']

    for sdir in STORAGE_DIRS:
        dname = path.join(folder, sdir)
        if not path.exists(dname):
            mkdir(dname)

#mk_db_structure(DATA_DIR)

# Sensors loaction on the body and order in the Exercice.X variable
SENSORS = ['Pied Droit', 'Pied Gauche', 'Ceinture', 'Tete']
ID_SENSORS = ['746', '748', '754', '672']

CAPTOR_PREFIX = '000_00340'

# Exercice.get_signal signal selsction descriptor
# DGCT for the sensor selection
# AR for the type selection (acceleration/ rotation)
# XYZ for the axis selection
DESC = {'D': 0, 'G': 6, 'C': 12, 'T': 18,
        'A': 0, 'R': 3,
        'X': 0, 'Y': 1, 'Z': 2}

# Gravitation constant to conver m/s² to g unit
G = 9.80665

from .database import Database
