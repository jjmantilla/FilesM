import numpy as np
import logging
from math import cos, sin
from os.path import join

from .. import DATA_DIR
from .. import SENSORS, N_SENSORS, CAPTOR_ID


class ChangeBase(object):
    """docstring for ChangeBase"""
    def __init__(self, fname):
        '''Recale les données dans le ref terestre pour les capteurs Xsens
        '''
        # import data
        print(fname)
        logging.debug(fname)
        #if 'Xsens' in fname:
        self.XsensImport(fname)
        #else:
        #    self.TCImport(fname)

        # return result immediately if algorithm builtin is called
        X = []
        for k in range(len(SENSORS)):
            X += [[], []]
            for t in range(self.time):
                X[2*k].append(np.dot(np.array(self.accelerometer[k][t]),
                              self.matrix[k][t]))
                X[2*k+1].append(np.dot(np.array(
                    self.gyroscope[k][t]), self.matrix[k][t]))
        X = np.array(X).swapaxes(1, 2)
        X = X.reshape((6*len(SENSORS), -1))
        self.X = X

        self.dataExport(fname)

    def dataExport(self, fname):
        N = len(self.X[0])
        for k, sensor in enumerate(SENSORS):
            fname_s = join(DATA_DIR, 'Data', fname.replace('PARTX', sensor))

            with open(fname_s, 'w') as f:
                f.write("Accelerometer_X;" + "Accelerometer_Y;" +
                        "Accelerometer_Z;" + "Gyroscope_X;" +
                        "Gyroscope_Y;" + "Gyroscope_Z;" +
                        "Sample Frequency;" + "\n")
                for i in range(N):
                    f.write(";".join(map(str, self.X[6*k:6*(k+1), i])) + ';')
                    if i == 0:
                        f.write(str(self.spf)+';')
                    f.write('\n')
            logging.info("Data correctly exported at %s", fname_s)
        return(None)

    def XsensImport(self, fname):
        '''import data for patient and recording specified
           for the Xsens captors
        '''

        # import the data from txt files
        data = []
        for ns in N_SENSORS:
            fpart = fname.replace('Xsens-PARTX.csv',
                                  CAPTOR_ID+ns+".txt")
            print('Loading ', fpart)
            data += [[]]

            # import data and tell so user
            with open(join(DATA_DIR, 'Raw', fpart)) as f:
                f.readline()
                spf = f.readline().split()[3]
                spf = float(spf.replace('Hz', ''))
                f.readline(), f.readline(), f.readline()
                for line in f.readlines():
                    data[-1].append([
                        float(x)
                        for x in line.strip('\n').split('\t') if x])
        logging.info('ChangeBase - Import data for %s',
                     fname.replace('PARTX.csv', ''))

        # homogenize data lengths
        # first index gives the sensor, second one gives access to one specific
        # parameter
        len_data = [len(ll) for ll in data]
        M = max(len_data)
        m = min(len_data)

        dr = abs(M - m) / m
        if m != M:
            logging.info("data length differ (of at most "
                         "{:6.2%}) from one sensor to another"
                         "".format(dr))
            for i in range(len(data)):
                data[i] = data[i][:m]

        self.time = m
        data = np.array(data)

        self.spf = spf
        # fill the dataExamen object with imported and cleaned data
        self.accelerometer = []
        self.gyroscope = []
        self.matrix = []

        for i in range(len(SENSORS)):
            self.accelerometer.append(data[i][:, 2:5])
            self.gyroscope.append(data[i][:, 8:11])
            self.matrix.append(data[i][:, 21:].reshape((-1, 3, 3)))

    def TCImport(self, fname):
        data = []
        mat = []
        for i in range(4, 0, -1):
            fname_s = fname.replace('PARTX.csv',
                                    'datas-{}.txt'.format(i))
            data += [[]]
            mat += [[]]

            # import data and tell so user
            with open(join(DATA_DIR, 'Raw', fname_s)) as f:
                f.readline(), f.readline(), f.readline(), f.readline()
                for line in f.readlines():
                    data[-1].append([
                        float(x.replace(',', '.'))
                        for x in line.strip('\n').split('\t') if x])
            with open(join(DATA_DIR, 'Raw',
                      fname_s.replace('datas', 'quats'))) as f:
                f.readline(), f.readline(), f.readline(), f.readline()
                for line in f.readlines():
                    ll = [float(x.replace(',', '.'))
                          for x in line.strip('\n').split('\t') if x]
                    ct, st = cos(ll[1]), sin(ll[1])
                    u = [ll[2], ll[3], ll[4]]
                    uu = np.outer(u, u)
                    uzu = np.cross(np.eye(3), u)
                    Rmat = ct*np.eye(3) + st*uzu + (1-ct)*uu
                    mat[-1].append(Rmat)

        logging.info('ChangeBase - Import data for %s',
                     fname.replace('PARTX.csv', ''))

        # homogenize data lengths
        # first index gives the sensor, second one gives access to one specific
        # parameter
        len_data = [len(vl) for vl in data]
        M = max(len_data)
        m = min(len_data)

        dr = abs(M - m) / m
        if m != M:
            logging.info("data length differ (of at most "
                         "{:6.2%}) from one sensor to another"
                         "".format(dr))
            for i in range(len(data)):
                data[i] = data[i][:m]
                mat[i] = mat[i][:m]

        self.time = m
        data = np.array(data)

        self.spf = 100
        # fill the dataExamen object with imported and cleaned data
        self.accelerometer = []
        self.gyroscope = []
        self.matrix = []

        for i in range(len(SENSORS)):
            self.accelerometer.append(data[i][:, 1:4])
            self.gyroscope.append(data[i][:, 7:10]/180*np.pi)
        self.matrix = mat
