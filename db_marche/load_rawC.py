from os.path import join as j
from csv import DictReader
import  matplotlib.pyplot as plt
import numpy as np
import pylab
DATA_DIR='/Users/jjmantilla/Documents/CMLA/Course/raw/'
SENSORS = ['Pied Droit', 'Pied Gauche', 'Ceinture', 'Tete']
N_SENSORS = ['2AC', '2B5', '2BB', '2BC']
CAPTOR_ID = '_00B41'


def load_raw_course(fn):
    scale_A=1
    scale_R=1
    delimiter='\t'
    X=[]
    fps = []
    for ns in N_SENSORS:
        fname = fn+ CAPTOR_ID+ns+'.txt'
        print(fname)
        res = []
        with open(j(DATA_DIR, fname)) as f:
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
    #assert((np.array(fps) == fps[0]).all())
    return X

def list():
    files=['MAN-Jua-Yo-5']#,
    #files=['VIE-Ali-Yo-0']
    files=['WAN-Dan-Yo-0']
    for r in files:
        print(r)
        Xr=load_raw_course(r)
        print(len(Xr[0][0]))
        plt.subplot(231);plt.plot(Xr[0][0][:].T)
        plt.subplot(232);plt.plot(Xr[0][1][:].T)
        plt.subplot(233);plt.plot(Xr[0][2][:].T)
        plt.subplot(234);plt.plot(Xr[0][3][:].T)
        plt.subplot(235);plt.plot(Xr[0][4][:].T)
        plt.subplot(236);plt.plot(Xr[0][5][:].T)
        pylab.show()
        plt.subplot(231);plt.plot(Xr[1][0][:].T)
        plt.subplot(232);plt.plot(Xr[1][1][:].T)
        plt.subplot(233);plt.plot(Xr[1][2][:].T)
        plt.subplot(234);plt.plot(Xr[1][3][:].T)
        plt.subplot(235);plt.plot(Xr[1][4][:].T)
        plt.subplot(236);plt.plot(Xr[1][5][:].T)
        pylab.show()
        plt.subplot(231);plt.plot(Xr[2][0][:].T)
        plt.subplot(232);plt.plot(Xr[2][1][:].T)
        plt.subplot(233);plt.plot(Xr[2][2][:].T)
        plt.subplot(234);plt.plot(Xr[2][3][:].T)
        plt.subplot(235);plt.plot(Xr[2][4][:].T)
        plt.subplot(236);plt.plot(Xr[2][5][:].T)
        pylab.show()
        plt.subplot(231);plt.plot(Xr[3][0][:].T)
        plt.subplot(232);plt.plot(Xr[3][1][:].T)
        plt.subplot(233);plt.plot(Xr[3][2][:].T)
        plt.subplot(234);plt.plot(Xr[3][3][:].T)
        plt.subplot(235);plt.plot(Xr[3][4][:].T)
        plt.subplot(236);plt.plot(Xr[3][5][:].T)
        pylab.show()

list()
