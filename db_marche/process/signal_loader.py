import numpy as np
import logging
from os.path import join

from .. import DATA_DIR, G
from .. import SENSORS, ID_SENSORS, CAPTOR_PREFIX

N_SENSORS = len(SENSORS)


logger = logging.getLogger("Database")


class SignalLoader(object):

    """ Class to load a signal from the raw captor file
    """

    def __init__(self, meta):
        self.meta = meta
        self.fname = meta['fname']
        self.Tcon = 'TCon' in meta['sensor']

        if self.Tcon:
            self._load_raw_Tcon()
        else:
            self._load_raw_Xsens()

        self._compute_recalibrate()
        self._correct_time_leap()

        self._swapaxes()
        #self._test_signal()

    def _load_raw_Xsens(self):
        """import data from Xsens raw file
        """
        scale_A = 1 / G
        scale_R = 180 / np.pi

        logger.info('Loading {}'.format(self.meta['id']))

        data = []
        for ids in ID_SENSORS:
            fpart = self.fname.replace('Xsens-PARTX.csv',
                                       CAPTOR_PREFIX + ids + ".txt")
            data += [[]]

            # import data and tell so user
            if fpart[0] != '/':
                fpart = join(DATA_DIR, 'Raw', fpart)
            with open(fpart) as f:
                # Skip the header of the raw file
                f.readline()
                spf = f.readline().split()[3]  # Get the sample rate
                spf = float(spf.replace('Hz', ''))
                assert spf == 100
                f.readline(), f.readline(), f.readline()

                # Read the data from the file
                for line in f.readlines():
                    data[-1].append([
                        float(x)
                        for x in line.strip('\n').split('\t') if x])

                data[-1] = np.array(data[-1])

        # fill the dataExamen object with imported and cleaned data
        self.accelerometer = []
        self.gyroscope = []
        self.matrix = []
        self.timestamps = []

        for i in range(N_SENSORS):
            self.timestamps.append(data[i][:, 0].astype(np.int64))
            self.accelerometer.append(data[i][:, 2:5]*scale_A)
            self.gyroscope.append(data[i][:, 8:11]*scale_R)
            self.matrix.append(data[i][:, 21:].reshape((-1, 3, 3)
                                                       ).swapaxes(1, 2))

        self.signal_sensor = [np.c_[acc, gyr]
                              for acc, gyr in zip(self.accelerometer,
                                                  self.gyroscope)]

    def _load_raw_Tcon(self):
        from csv import DictReader
        scale_A = 1
        scale_R = 1

        logger.info('Loading {}'.format(self.meta['id']))

        data, self.matrix, self.timestamps = [], [], []
        for i in range(4, 0, -1):
            fname_s = self.fname.replace('PARTX.csv',
                                         'datas-{}.txt'.format(i))
            data_s = []
            self.matrix += [[]]
            self.timestamps += [[]]

            with open(join(DATA_DIR, 'Raw', fname_s)) as f:
                f.readline()
                f.readline()
                f.readline()
                for row in DictReader(f, delimiter='\t'):
                    self.timestamps[-1] += [round(get_num(row['[time]'])*100)]
                    data_s.append([get_num(row['[accX]']),
                                   get_num(row['[accY]']),
                                   get_num(row['[accZ]']),
                                   get_num(row['[gyroX]']),
                                   get_num(row['[gyroY]']),
                                   get_num(row['[gyroZ]'])])

            data += [np.array(data_s)]

            with open(join(DATA_DIR, 'Raw',
                           fname_s.replace('datas', 'quats'))) as f:
                f.readline(), f.readline(), f.readline()
                for row in DictReader(f, delimiter='\t'):
                    w = get_num(row['[w]'])
                    u = np.array([get_num(row['[w]']), get_num(row['[x]']),
                                  get_num(row['[y]']), get_num(row['[z]'])])
                    nu = np.sqrt((u*u).sum())
                    u /= nu
                    w, x, y, z = u[0], u[1], u[2], u[3]
                    Rmat = [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]
                    # ct, st = cos(w), sin(w)
                    # nu = np.sqrt((u*u).sum())
                    # u = u/nu
                    # uu = np.outer(u, u)
                    # uzu = np.cross(np.eye(3), u)
                    # Rmat = ct*np.eye(3) + st*uzu + (1-ct)*uu
                    assert np.isclose(np.trace(Rmat)+1, 4*w*w, 1e-3)
                    #Rmat = -np.array(Rmat)#*[[1, 1 ,-1],[1, 1, -1],[1, 1 ,-1]]
                    P_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    self.matrix[-1].append(P_mat.dot(Rmat))

        # fill the dataExamen object with imported and cleaned data
        self.accelerometer = []
        self.gyroscope = []

        for i in range(N_SENSORS):
            self.accelerometer.append(data[i][:, :3]*scale_A)
            self.gyroscope.append(data[i][:, 3:]*scale_R)

        self.signal_sensor = [np.c_[acc, gyr]
                              for acc, gyr in zip(self.accelerometer,
                                                  self.gyroscope)]

    def _compute_recalibrate(self):
        # return result immediately if algorithm builtin is called
        acc_rec = [np.array([m.dot(val) for val, m in zip(sig, matrix)])
                   for sig, matrix in zip(self.accelerometer, self.matrix)]
        gyr_rec = [np.array([m.dot(val) for val, m in zip(sig, matrix)])
                   for sig, matrix in zip(self.gyroscope, self.matrix)]

        self.signal_earth = [np.c_[acc, gyr]
                             for acc, gyr in zip(acc_rec, gyr_rec)]

    def _correct_time_leap(self):
        T_b = []
        time_leaps = []
        time_indexes = []
        data_sensor, data_earth = [], []
        for ns in range(N_SENSORS):
            time = self.timestamps[ns]
            # Make sure the packet conter is increasing
            if time[0] > time[-1]:
                m_time, M_time = time[0], max(time)
                time = [t if t >= m_time else t+M_time+1 for t in time]
            assert len(time) == len(set(time))
            time_indexes += [time]
            T_b += [[time[0], time[-1]]]

            # Interpolate missing values
            ss = np.empty((time[-1] - time[0] + 1, 6))
            se = np.empty((time[-1] - time[0] + 1, 6))
            sig_sensor = self.signal_sensor[ns]
            sig_earth = self.signal_earth[ns]
            ss[0] = sig_sensor[0]
            se[0] = sig_earth[0]
            t0 = time[0]
            time_leap = []
            for i, (t, t1) in enumerate(zip(time[:-1], time[1:])):
                if t1-t > 1:
                    K = t1-t
                    for k in range(K):
                        # Interpolate the values
                        ss[t-t0+k+1] = ((K-1-k)/K * sig_sensor[i] +
                                        (k+1)/K*sig_sensor[i+1])
                        se[t-t0+k+1] = ((K-1-k)/K * sig_earth[i] +
                                        (k+1)/K * sig_earth[i+1])
                    time_leap += [[i, K]]
                else:
                    ss[t1-t0] = sig_sensor[i+1]
                    se[t1-t0] = sig_earth[i+1]
            data_sensor += [ss]
            data_earth += [se]
            time_leaps += [time_leap]

        # Remove extra value at begin and end
        T_m, T_M = np.max(T_b, axis=0)[0], np.min(T_b, axis=0)[1]
        data_sensor = np.c_[[x[T_m-t0:T_M-t1 if T_M-t1 < 0 else None]
                             for x, (t0, t1) in zip(data_sensor, T_b)]
                            ]
        data_earth = np.c_[[x[T_m-t0:T_M-t1 if T_M-t1 < 0 else None]
                            for x, (t0, t1) in zip(data_earth, T_b)]
                           ]

        data_sensor = data_sensor.swapaxes(1, 2).reshape(24, -1)
        data_earth = data_earth.swapaxes(1, 2).reshape(24, -1)

        self.data_earth, self.data_sensor = data_earth, data_sensor
        self.time_leaps, self.time_indexes = time_leaps, time_indexes

    def _swapaxes(self):

        if not self.Tcon:
            return

        data_sensor = self.data_sensor
        sensor_r = np.empty(self.data_sensor.shape)
        # Tcon -> Xsens for feets
        sensor_r[:12:3] = -data_sensor[1:12:3]
        sensor_r[1:12:3] = data_sensor[:12:3]
        sensor_r[2:12:3] = data_sensor[2:12:3]

        # Tcon -> Xsens for belt
        sensor_r[12:18:3] = data_sensor[13:18:3]
        sensor_r[13:18:3] = -data_sensor[12:18:3]
        sensor_r[14:18:3] = data_sensor[14:18:3]

        # Tcon -> Xsens for head
        sensor_r[18::3] = -data_sensor[18::3]
        sensor_r[19::3] = -data_sensor[19::3]
        sensor_r[20::3] = data_sensor[20::3]

        self.data_sensor = sensor_r

    def _test_signal(self):
        """Assert the quality of the acquired signal by verifying:

        - There are not too many lost samples
        - The signal is long enough
        - The protocol have been respected
        """
        timestamps = self.timestamps
        duration = [len(ts) for ts in timestamps]
        max_diff = [np.max(np.abs(np.diff(ts))) for ts in timestamps]

        if np.max(max_diff) > 10:
            raise RejectedExerciseError("Too many samples have been lost")

        if np.min(duration) < 1e3:
            raise RejectedExerciseError("One signal is too short")

        self.quality_signal = np.std(duration)
        self._test_protocol()

    def _test_protocol(self):
        """Test if the signal is following the defined protocol

        - Check for sensor inversion
        - Check for movement during the first 4 seconds
        """
        thr = 100
        d = 4
        Fs = 100
        data_sensor = self.data_sensor

        # Do not test the head as the patient might move it a lot
        for i, sensor in enumerate(['right foot', 'left foot', 'lower back']):

            # Test for movment in the first 4s
            mov = np.sum([sig[:d*Fs].std()
                          for sig in data_sensor[3+i*6:(i+1)*6]])
            if mov > thr:
                raise RejectedExerciseError("Subject has moved during the 4"
                                            " first seconds {}".format(sensor))

        if ((np.mean(data_sensor[0, 0:d*Fs]) < 0) |
            (np.abs(np.mean(data_sensor[1, 0:d*Fs])) > 0.5) |
                (np.mean(data_sensor[2, 0:d*Fs]) < 0.5)):
            import matplotlib.pyplot as plt
            plt.plot(data_sensor[:3].T)
            plt.show()
            raise RejectedExerciseError("Right foot sensor is not correctly "
                                        "positioned.")

        if ((np.mean(data_sensor[0+6, 0:d*Fs]) < 0) |
            (np.abs(np.mean(data_sensor[1+6, 0:d*Fs])) > 0.5) |
                (np.mean(data_sensor[2+6, 0:d*Fs]) < 0.5)):
            raise RejectedExerciseError("Left foot sensor is not correctly "
                                        "positioned.")

        if ((np.mean(data_sensor[0+12, 0:d*Fs]) < 0.5) |
            (np.mean(data_sensor[1+12, 0:d*Fs]) > 0.5) |
                (np.mean(data_sensor[2+12, 0:d*Fs]) > 0.5)):
            raise RejectedExerciseError("Lower back sensor is not correctly "
                                        "positioned")


class RejectedExerciseError(Exception):

    def __init__(self, reason):
        super(RejectedExerciseError, self).__init__(reason)
        self.reason = reason

    def __str__(self):
        return "RejectedExerciseError: {}".format(self.reason)


def get_num(v):
    try:
        return float(v)
    except ValueError:
        return float(v.replace(',', '.'))
