import numpy as np
from os.path import join as j
from csv import DictReader
import logging

from . import DATA_DIR, G
from . import SENSORS, N_SENSORS, CAPTOR_ID

np.seterr(all='raise')


def load_csv(fn, delimiter=';'):
    X = []
    fps = []

    # Scaling to get comparable units Xsens/Tcon
    scale_A = 1 / G
    scale_R = 180 / np.pi
    if 'TCon' in fn:
        scale_R = scale_A = 1

    # Read files for all sensors
    for s in SENSORS:
        fname = fn.replace('PARTX', s)
        with open(j(DATA_DIR, 'Data', fname)) as f:
            # Parse categorie name (except sampling rate)
            res = []
            for i, row in enumerate(DictReader(f, delimiter=delimiter)):
                if i == 0:
                    fps += [float(row['Sample Frequency'])]
                res.append([float(row['Accelerometer_X'])*scale_A,
                            float(row['Accelerometer_Y'])*scale_A,
                            float(row['Accelerometer_Z'])*scale_A,
                            float(row['Gyroscope_X'])*scale_R,
                            float(row['Gyroscope_Y'])*scale_R,
                            float(row['Gyroscope_Z'])*scale_R])
            X += [np.transpose(res)]

    X = np.c_[X].reshape((24, -1)).astype(float)
    Xr = np.empty(X.shape)
    if 'TCon' in fn:
        Xr[:12:3] = -X[1:12:3]
        Xr[1:12:3] = X[:12:3]
        Xr[2:12:3] = X[2:12:3]

        Xr[12::3] = X[13::3]
        Xr[13::3] = -X[12::3]
        Xr[14::3] = X[14::3]
    assert((np.array(fps) == fps[0]).all())
    return Xr, fps[0]


def load_csv_raw(fn, delimiter='\t'):
    X = []
    lr = []

    # Scaling to get comparable units Xsens/Tcon
    scale_A = 1 / G
    scale_R = 180 / np.pi

    # Read file of values
    for ns in N_SENSORS:
        fname = fn.replace('Xsens-PARTX.csv', CAPTOR_ID+ns+'.txt')
        if '/' not in fname:
            fname = j(DATA_DIR, 'Raw', fname)
        res = []
        with open(fname) as f:
            f.readline()
            l_spr = f.readline()
            f.readline()
            f.readline()

            # Parse categorie name (except sampling rate)
            for row in DictReader(f, delimiter=delimiter):
                res.append([float(row['Acc_X'])*scale_A,
                            float(row['Acc_Y'])*scale_A,
                            float(row['Acc_Z'])*scale_A,
                            float(row['Gyr_X'])*scale_R,
                            float(row['Gyr_Y'])*scale_R,
                            float(row['Gyr_Z'])*scale_R])
        X += [np.transpose(res)]
        lr += [len(res)]
    dr = (max(lr)-min(lr))/min(lr)
    if 0.03 > dr > 0.01:
        logging.info("{}: Data length differ of at most {:5.2%}".format(
            fn.replace('-PARTX.csv', ''), dr))
    elif dr > 0:
        logging.debug("{}: Data length differ of at most {:5.2%}".format(
            fn.replace('-PARTX.csv', ''), dr))
    assert (dr < 0.03), 'Data length differs of more than 3%'

    X = np.c_[[x[:, :min(lr)] for x in X]].reshape(-1, min(lr))
    fps = float(l_spr.replace('Hz', '').replace('// Update Rate: ', ''))
    return X, fps


def get_num(v):
    try:
        return float(v)
    except ValueError:
        return float(v.replace(',', '.'))


def load_csv_technoconcept(fn, delimiter='\t'):
    X = []
    lr = []

    # Read file
    for i in range(4, 0, -1):
        fname = fn.replace('PARTX.csv', 'datas-{}.txt'.format(i))
        if '/' not in fname:
            fname = j(DATA_DIR, 'Raw', fname)
        res = []
        with open(fname) as f:
            f.readline()
            f.readline()
            f.readline()
            for row in DictReader(f, delimiter=delimiter):
                res.append([get_num(row['[accX]']),
                            get_num(row['[accY]']),
                            get_num(row['[accZ]']),
                            get_num(row['[gyroX]']),
                            get_num(row['[gyroY]']),
                            get_num(row['[gyroZ]'])])
        X += [np.transpose(res)]
        lr += [len(res)]

    # Verify that the data length are not to different
    dr = (max(lr)-min(lr))/min(lr)
    if 0.03 > dr > 0.01:
        logging.info("{}: Data length differ of at most {:5.2%}".format(
            fn.replace('-TCON-PARTX.csv', ''), dr))
    elif dr > 0:
        logging.debug("{}: Data length differ of at most {:5.2%}".format(
            fn.replace('-TCON-PARTX.csv', ''), dr))
    assert (dr < 0.03), 'Data length differs of more than 3%'

    X = np.c_[[x[:, :min(lr)] for x in X]].reshape((-1, min(lr)))
    Xr = np.empty(X.shape)

    # Inverse axis to match Xsens ones
    Xr[:12:3] = -X[1:12:3]
    Xr[1:12:3] = X[:12:3]
    Xr[2:12:3] = X[2:12:3]

    Xr[12::3] = X[13::3]
    Xr[13::3] = -X[12::3]
    Xr[14::3] = X[14::3]
    return Xr, 100
