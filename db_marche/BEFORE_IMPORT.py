
import numpy as np
from db_marche import Database
from matplotlib import pyplot as plt


# Je suppose que j'ai en entrée :
#     - data_sensor un array contenant 24 listes (pouvant être de tailles différentes)
#     - timestamp_sensor un array contenant 4 listes (un par capteur, pouvant être de tailles différentes)
#     - Fs la fréquence d'échantillonnage


# ETAPE 1 : LES PROBLEMES DE TAILLES ET DE TEMPORALITE
# Renvoie <0 si l'exercice est rejeté
#         0.0 si tout est parfait
#         >0 si il y a un probleme, dans ce cas plus cela est élevé, plus cela est mauvais
def check_time(timestamp_sensor,Fs):
    if Fs!=100:
        print("Sampling frequency is not correct");
        return -1

    N_time=len(timestamp_sensor)
    duration=[]
    max_diff=[]
    for i in range(N_time):
        duration.append(len(timestamp_sensor[i]))
        max_diff.append(np.max(np.abs(np.diff(timestamp_sensor[i]))))

    if np.max(max_diff)>10*Fs:
        print("Too many samples have been lost");
        return -2

    if np.min(duration)<10*Fs:
        print("One signal is too short");
        return -3

    return np.std(duration)


# ETAPE 2 : LES PROBLEMES DE CAPTEURS ET DE RESPECT DE PROTOCOLE
# Renvoie <0 si l'exercice est rejeté
#         0.0 si tout est parfait
#         >0 si il y a un probleme, dans ce cas plus cela est élevé, plus cela est mauvais
def check_protocol(data_sensor,Fs):
    thr=100
    d=4
    x1=np.std(data_sensor[3,0:d*Fs])+np.std(data_sensor[4,0:d*Fs])+np.std(data_sensor[5,0:d*Fs])
    if x1>thr:
        print("Subject has moved during the 4 first seconds (right foot)")
        return -1
    x2=np.std(data_sensor[3+6,0:d*Fs])+np.std(data_sensor[4+6,0:d*Fs])+np.std(data_sensor[5+6,0:d*Fs])
    if x2>thr:
        print("Subject has moved during the 4 first seconds (left foot)")
        return -2
    x3=np.std(data_sensor[3+12,0:d*Fs])+np.std(data_sensor[4+12,0:d*Fs])+np.std(data_sensor[5+12,0:d*Fs])
    if x3>thr:
        print("Subject has moved during the 4 first seconds (lower back)")
        return -3

    if ((np.mean(data_sensor[0,0:d*Fs])<0) | (np.abs(np.mean(data_sensor[1,0:d*Fs]))>0.5) | (np.mean(data_sensor[2,0:d*Fs])<0.5)):
        print("Right foot sensor is not correctly positioned")
        return -4

    if ((np.mean(data_sensor[0+6,0:d*Fs])<0) | (np.abs(np.mean(data_sensor[1+6,0:d*Fs]))>0.5) | (np.mean(data_sensor[2+6,0:d*Fs])<0.5)):
        print("Left foot sensor is not correctly positioned")
        return -5

    if ((np.mean(data_sensor[0+12,0:d*Fs])<0.5) | (np.mean(data_sensor[1+12,0:d*Fs])>0.5) | (np.mean(data_sensor[2+12,0:d*Fs])>0.5)):
        print("Lower back sensor is not correctly positioned")
        return -6

    return 0

# Je me crée des données pour tester le code
db = Database()
N = db.n_samples
var1=[]
var2=[]
capteurs=[]
nom=[]
for j in range(N):
    ex=db.load_data_object(j)
    data_sensor=ex.data_sensor
    # Correction a la main pour les TCon (non necessaire par la suite)
    # if ex.meta['sensor'] == "TCon":
    #     data_sensor[4]=-data_sensor[4]
    #     data_sensor[1]=-data_sensor[1]
    #     data_sensor[3]=-data_sensor[3]
    #     data_sensor[0]=-data_sensor[0]
    #     data_sensor[4+6]=-data_sensor[4+6]
    #     data_sensor[1+6]=-data_sensor[1+6]
    #     data_sensor[3+6]=-data_sensor[3+6]
    #     data_sensor[0+6]=-data_sensor[0+6]
    # Creation manuelle d'un faux timestamp
    timestamp_sensor=[]
    for i in range(4):
        t=np.arange(len(data_sensor[i*6]))/100
        timestamp_sensor.append(t)
    timestamp_sensor=np.array(timestamp_sensor)
    Fs=100
    print("%d - %s - %s"%(j,ex.meta["id"],ex.meta["sensor"]))
    capteurs.append(ex.meta["sensor"])
    nom.append(ex.meta["id"])
    var1.append(check_time(timestamp_sensor,Fs))
    var2.append(check_protocol(data_sensor,Fs))

erreurs=np.where((np.array(var1)<0) | (np.array(var2)<0))
erreurs1=np.where((np.array(var1)<0))
erreurs2=np.where((np.array(var2)==-1) | (np.array(var2)==-2) | (np.array(var2)==-3))
erreurs3=np.where((np.array(var2)==-4) | (np.array(var2)==-5) | (np.array(var2)==-6))

print('\n')
print("La base contient %d exercices rejetés sur un total de %d exercices"%(len(erreurs[0]),N))
print("- %d exercices ont des problèmes de taille ou de temporalité"%(len(erreurs1[0])))
print("- %d exercices ont des problèmes de respect de protocole (mouvement durant les 4 premieres secondes)"%(len(erreurs2[0])))
print("- %d exercices ont des problèmes d'orientation de capteur"%(len(erreurs3[0])))

