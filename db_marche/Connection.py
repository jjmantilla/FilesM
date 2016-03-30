import  matplotlib.pyplot as plt
import numpy as np
import pylab
from db_marche.exercise import Exercise
from db_marche import database
from db_marche.process.ChangeBase import ChangeBase


from Quaternion import Quat
u=np.array([0.465391,	-0.711934, 0.438905,	-0.289695])
nu = np.sqrt((u*u).sum())
u /= nu

w, x, y, z = u[0], u[1], u[2], u[3]
Rmat = [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]
assert np.isclose(np.trace(Rmat)+1, 4*w*w, 1e-3)
Rmat = -np.array(Rmat)
print(Rmat)
u=np.array([-0.711934, 0.438905,-0.289695,0.465391,])
#               x         y          z       w
q = Quat(u)
matrix=q._get_transform()
print(matrix)


db = database.Database(debug=1)
ex = db.get_data(fname="CHA-Nic-YO-11-TCon-PARTX.csv")[0]
print("tot ",len(ex.data_earth))
d=4
Fs=100
print(len(ex.data_sensor[3,0:d*Fs]))
import db_marche.process.Segmentation as seg

s = seg.Segmentation(ex, level=0.2)
s.fit()
print(s.seg_from_labels())

#ex1= Exercise(db._process_name(fname="MAN-Jua-YO-1-Xsens-PARTX.csv"))

#print(db.meta)
#cb=ChangeBase(fname="MAN-Jua-YO-1-Xsens-PARTX.csv")
#e1 = db.load_data_object(fname="MAN-Jua-YO-1-Xsens-PARTX.csv")#


#e1 = database._Database.get_data(self=database,fname="MAN-Jua-YO-1-XSens-PARTX")
#print("number of exercises in the BD: ",len(list_exercises))

##############################################################

#exercise=list_exercises[0]
#e=exercise.get_signal("DRX").T
#e1=exercise.get_signal("rDGX").T
#print(exercise.fname)
# from db_marche.load_csv import load_csv
# X2=load_csv("CHA-Nic-YO-11-TCon-PARTX.csv")
# #for i in X2[0][23].T:
# #    print(i)
# print("Recalés: ")
# print(e[:20].flatten())
# #print("Raw: ")
# #print(e1[:20].flatten())
# x_0 = 0.000214012077
# #x_0 =0.000896783945
# x_0 =-0.000390898786999
#
# #ee = e + x_0 - e[0]
#
#
# #print(ee[:20].flatten())
# #print(abs(ee-e)[:20].flatten())
# plt.plot(e)
# #pylab.show()
#
# T = exercise.X.shape[1]
# # t = np.arange(T)/100
# # plt.subplot(321);plt.plot(t,exercise.get_signal("DRX")[0])
# # plt.subplot(322);plt.plot(t,exercise.get_signal("DAX")[0])
# # plt.subplot(323);plt.plot(t,exercise.get_signal("DRY")[0])
# # plt.subplot(324);plt.plot(t,exercise.get_signal("DAY")[0])
# # plt.subplot(325);plt.plot(t,exercise.get_signal("DRZ")[0])
# # plt.subplot(326);plt.plot(t,exercise.get_signal("DAZ")[0])
# # pylab.show()
#
# plt.figure('Segmentation')
# seg = exercise.seg_annotation
# print(' %f %f %f %f' % (seg[0],seg[1],seg[2],seg[3]))
# plt.subplot(121); plt.plot(exercise.CAZ.T[seg[0]:seg[1]])  # Forward walk
# plt.subplot(122); plt.plot(exercise.CAZ.T[seg[2]:seg[3]])  # Backward walk
#
# plt.figure('Steps')
# list_step_right = exercise.steps_annotation[0]             # Step annotation for the right foot
# print(len(list_step_right))
# step4 = list_step_right[1]
#
# plt.plot(exercise.CAZ.T[step4[0]-10:step4[1]+10])
#
# #exercise.load_feat(exercise)
# from db_marche.process.Feature import Feature
# #Feature.compute(exercise,exercise)
# pylab.show()
# from db_marche.process import step_detection
# #compute_steps(exercise)