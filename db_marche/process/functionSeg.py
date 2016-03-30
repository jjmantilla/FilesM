# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .SegStep import *
from .Pattern import Bibliotheque
from .calibration import *


def simCos(x, y):
    ''' Return the cosine similarity between two vectors, if the length
        of the vectors are different, it returns the cosine similarity
        on the common domain

        Parameters:
        ----------
        x: float list
        y: float list
    '''
    l1 = len(x)
    l2 = len(y)
    l = min(l1, l2)

    u = x[0:l - 1]
    v = y[0:l - 1]

    z = 1. / np.linalg.norm(v) * 1. / np.linalg.norm(u) * np.dot(u, v)
    return z


def testBruit(fen, seuil):
    '''Return a boolean which is true if the norm of the signal is under a threshold on a window.
       This function is temporary since the detection of the first walking phase is pending.
       It enables the algorithm to lower the number of operations.

       Parameters:
       ----------
        fen: float list
            Window
        seuil: float
            Threshold
    '''
    y = True
    for i in range(len(fen)):
        if abs(fen[i]) > seuil:
            y = False
            break

    return y


def testBruit2(fen):
    ''' Return a value quantifying the noise on a given signal, with a L1 norm

        Parameters:
        ----------

        fen: float list
             part of a signal
    '''

    x = 0
    for i in fen:
        x = x + abs(i)

    return x / len(fen)


def inDemiTour1(pos, dmt):
    ''' Return a boolean which is true if the current position is within the u-turn.

        Parameters:
        ----------

        pos: int
             Current position
        dmt: list of 3 elements:
            - beginning of the u-turn
            - end of the u-turn
            - end of the second walking phase
    '''
    if (pos > dmt[0] and pos < dmt[1]) or pos > dmt[2]:
        return True
    else:
        return False


def inDemiTour2(pos, p, dmt):
    ''' Return a boolean which is true if the end of a step is within the u-turn.

        Parameters:
        ----------
        pos: int
             Current position
        p: Pattern
        dmt: list of 3 elements:
            - beginning of the u-turn
            - end of the u-turn
            - end of the second walking phase
    '''
    if pos + len(p._vect) > dmt[0] and pos + len(p._vect) < dmt[1] or pos + len(p._vect) > dmt[2]:
        return True
    else:
        return False


def inDemitour3(pos, p, dmt):
    ''' Return a boolean which is true if the u-turn is included in a single step

    Parameters:
    ----------
    pos: int
         Current position
    p: Pattern
    dmt: list of 3 elements:
        - beginning of the u-turn
        - end of the u-turn
        - end of the second walking phase
    '''

    if (pos <= dmt[0]) & (pos + len(p._vect) >= dmt[1]):
        return True
    else:
        return False


def matchSig(coord, donnees):
    '''Return the signal which has to be taken in a account in the matching of a pattern.

       Parameters:
       ----------
       coord= str
            Signal from which the pattern has been selected.
       donnees= donnees
            Donnees object corresponding to the six signals of a captor.
    '''
    if coord == 'ax':
        return donnees._acc[0]
    elif coord == 'az':
        return donnees._acc[2]
    else:
        return donnees._gyr[1]


def miseAJour(donneesPied, bibliotheque, bibliothequeString, newPas, filename):
    '''Complete the pattern library, if an unknown pattern is detected.
       Possible if and only if the selected version of the algorithm is interactive.

       Parameters:
       ----------
       donneesPied: donnees object
                   corresponding to a foot sensor
       bibliotheque: Bibliotheque object
                   correspond to a pattern library
       bibliothequeString: str
                   filename in which the library has to be written
       newPas: list
                   new step detected (see parcours for details)
       filename: str
                   filename from which the new step is coming
    '''
    from .Pattern import Pattern

    newPattern = Pattern()
    newPattern._filename = filename
    newPattern._score = 0
    newPattern._vect = donneesPied._acc[0][newPas[0]:newPas[0]+newPas[2]]
    newPattern._coord = 'ax'
    bibliotheque.append(newPattern)

    newPattern1 = Pattern()
    newPattern1._filename = filename
    newPattern1._score = 0
    newPattern1._vect = donneesPied._acc[2][newPas[0]:newPas[0]+newPas[2]]
    newPattern1._coord = 'az'
    bibliotheque.append(newPattern1)

    newPattern2 = Pattern()
    newPattern2._filename = filename
    newPattern2._score = 0
    newPattern2._vect = donneesPied._gyr[1][newPas[0]:newPas[0]+newPas[2]]
    newPattern2._coord = 'gy'
    bibliotheque.append(newPattern2)

    bibliotheque.save(bibliothequeString)


def parcours(donneesPied, donneesCeint, longFen=160, seuilBruit=1, seuilCor=0.78,
             posIni=300, posFin=-1, incr=False, calibr=True):
    '''Return three lists of step positions [position, number of the matching pattern,
                                             length of the step, coord of the matching pattern,
                                             vector of the step detected on this very coord,
                                             list of similarities to the step library].
       The first one is the total step list
       The second one is the first way walk
       The third one is the second way walk

    Parameters:
    ---------
    donneesPied: donnees object
                donnees of a foot captor
    donneesCeint: donnees object
                donnees of the waist captor
    seuilCor: float, optional(default: 0.8)
                threshold of the cosine similarity
    seuilBruit: float, optional(default: 1)
                threshold oh the testBruit function
    longFen: int (default: 160)
                length of the window of the testBruit function
    posIni: int (default: 300)
                beginning of the sliding position
    posFin: int (default: -1)
                end of the sliding position
    incr: boolean (default= False)
                increment the score of the matching patterns
    calibr: boolean (default= True)
                once a step is detected, the calibration is done
    '''

    # Chargement de la bibliothèque
    if donneesPied._capteur == 'Pied Gauche':
        bibliothequeString = 'listPatternsG'
        bibliotheque = Bibliotheque('listPatternsG.pickle')

    else:
        bibliothequeString = 'listPatternsD'
        bibliotheque = Bibliotheque('listPatternsD.pickle')

    # Position jusqu'où le parcours doit se faire
    if posFin == -1:
        posFin = donneesPied.getTemps()

    # Postion où le parcours doit commencer
    pos = posIni

    # Initialisation de la liste de pas aller
    pasAller = []
    # Initialisation de la liste de pas retour
    pasRetour = []
    # Initialisation de la liste de pas
    pas = []

    # Definition du bruit pour la calibration

    m, std = donneesPied.getNoise()

    # Numéro du pas courant
    num = 0

    # Liste de segmentation de la marche
    dmt = donneesCeint.getDemiTours()

    while pos < posFin:
        #Test demi-tour
        if inDemiTour1(pos, dmt):
            pos += 1
        else:
            # Test bruit
            if (testBruit(donneesPied._acc[0][pos:pos + longFen], seuilBruit) and
                    testBruit(donneesPied._acc[2][pos:pos + longFen], seuilBruit) and
                    testBruit(donneesPied._gyr[1][pos:pos + longFen], seuilBruit)):
                pos = pos + longFen

            else:
                noMatch = True
                # A la position courrante on veut déterminer si un pattern match:

                for i in range(len(bibliotheque.data)):
                    p = bibliotheque.data[i]

                    if len(p._vect) > posFin-pos:
                        pass
                    else:
                        # Test demi-tour
                        if inDemiTour2(pos, p, dmt):
                            pass
                        elif inDemitour3(pos, p, dmt):
                            pass
                            # Test de similarité
                        elif simCos(p._vect,
                                    matchSig(p._coord,
                                             donneesPied)[pos:pos + len(p._vect)]) > seuilCor:
                            c = []
                            for j in range(len(bibliotheque.data)):
                                q = bibliotheque.data[j]
                                c.append(simCos(q._vect, matchSig(q._coord,
                                                                  donneesPied)[pos:pos + len(q._vect)]))

                            if incr:
                                p.increment()
                            noMatch = False
                            if calibr:
                                deb1, fin1 = calibrage(donneesPied._acc[2], m, std, pos, pos+len(p._vect)-5)
                                pas.append([deb1, i, fin1-deb1,
                                            p._coord,
                                            matchSig(p._coord,
                                                     donneesPied)[deb1:fin1], c])
                                pos = fin1 + 1
                                num = num + 1

                            else:
                                pas.append([pos, i, len(p._vect),
                                            p._coord,
                                            matchSig(p._coord,
                                                     donneesPied)[pos:pos + len(p._vect)], c])
                                pos = pos + len(p._vect)
                                num = num + 1
                            break

                if noMatch:
                    pos = pos + 1

    # On remplit les pas aller et retour
    for i in range(len(pas)):
        if pas[i][0] < dmt[0]:
            pasAller.append(pas[i])
        else:
            pasRetour.append(pas[i])

    if incr:
        bibliotheque.save(bibliothequeString+'.pickle')

    return pas, pasAller, pasRetour


def testAmp(donneesPied, pas, coeff=5., version=2.):
    '''Targets the suspect steps in a step list if the max antero posterior acceleration amplitude
       is coeff times smaller then the median maximum amplitude of the step.
       If the version is interactive, the user has to decide if the suspect step has to be kept.
       Otherwise the step is deleted.

       Parameters:
       ----------
       donneesPied: donnees object
                    signals of a foot captor
       pas: list
            step list (see parcours for details on the step list structure)
       coeff: float (defaul: 5.)
            ratio of the amplitude under which a step is suspect
       version: 1. or 2.
            version of the algorithm: 1: interactive, 2: automatic
    '''

    #Définition de la médiane des amplitudes max:
    listeMax = []
    pasX = [[pas[j][0], pas[j][1], pas[j][2], pas[j][3],
            donneesPied._acc[0][pas[j][0]:pas[j][0] + pas[j][2]]]
            for j in range(len(pas))]

    for i in range(len(pasX)):
        listeMax.append(max([max(pasX[i][4]), -min(pasX[i][4])]))
    medianMax = np.median(np.array(listeMax))

    compt = 0
    while compt <= len(pas) - 1:
        if max([max(pasX[compt][4]), -min(pasX[compt][4])]) < (medianMax / coeff):

            if version == 1.:
                plt.figure('step')
                plt.ion()
                plt.plot(donneesPied._acc[0])
                plt.plot([pas[compt][0], pas[compt][0]], [-10, 10], 'r', linewidth=2.0)
                plt.plot([pas[compt][0] + pas[compt][2], pas[compt][0] + pas[compt][2]],
                         [-10, 10], 'm:', linewidth=2.0)
                plt.xlabel('Temps 1:100 sec')
                plt.ylabel('Acceleration m^2.sec^-1')
                plt.legend(
                    ['Acceleration axe antero posterieur', 'Debut du pas douteux',
                     'Fin du pas douteux'], loc='upper left', prop={'size': 5})
                plt.draw()
                plt.show()

                reponse = input("retirer le pas? y/n:")
                print(reponse)
                if reponse == "y":
                    del(pas[compt])
                    del(pasX[compt])
                else:
                    if reponse == "n":
                        compt = compt + 1

            else:
                if version == 2.:
                    del(pas[compt])
                    del(pasX[compt])

        else:
            compt = compt + 1


def testMediane(donnees, donneesCeint, pasAR, coeff=1.3, version=2.,
                bibliotheque=None, bibliothequeString=None):
    '''Targets if there is a suspect lapse of time between two steps,
       and verify if a step has been forgotten either automatically or by asking the user.

       Parameters:
       ----------
       donnees: donnees object
                signals of a foot captor
       donneesCeint: donnees object
                signals of the waist captor
       pasAR:   step list (see parcours for details)
                the list has to be either from the first way walking or the second one
       coeff: float (default: 1.3)
                if the lapse of time between two steps is coeff times bigger than the
                median time between two steps then the intervall is suspect
       version: 1. or 2. (default: 2.)
                ***if version = 1. (interactive)
                a plot is displayed and the user enters manually the beginning and the end
                of a forgotten step
                plus with this version the user can add a pattern into the pattern library
                ***if version = 2. (automatic)
                then a new parcours is executed between the end of the first step and
                the beggining of the second step
       bibliotheque:
                pattern library (automatically determined) (leave None)
       bibliothequeString:
                filename of the pattern library (automatically determined) (leave None)
    '''
    compteur = 0
    compt = 0

    while compt < len(pasAR) - 1:
        debutPas = []
        for i in range(len(pasAR)):
            debutPas.append(pasAR[i][0])
        diffPas = np.diff(np.array(debutPas))

        med = np.median(diffPas)
        if abs(diffPas[compt]) > coeff * med:

            if version == 1.:
                plt.ion()
                plt.figure('step')

                plt.plot(donnees._acc[0])
                plt.plot([pasAR[compt][0], pasAR[compt][0]],
                         [-10, 10], 'k', linewidth=3.0)
                plt.plot([pasAR[compt + 1][0], pasAR[compt + 1][0]],
                         [-10, 10], 'k', linewidth=3.0)
                plt.plot([pasAR[compt][0] + pasAR[compt][2], pasAR[compt][
                         0] + pasAR[compt][2]], [-10, 10], 'r:', linewidth=3.0)
                plt.plot([pasAR[compt + 1][0] + pasAR[compt + 1][2], pasAR[compt + 1][
                         0] + pasAR[compt + 1][2]], [-10, 10], 'r:', linewidth=3.0)
                plt.draw()
                plt.show()

                probleme = input('Probleme? y/n:')
                if probleme == 'y':

                    if version == 1.:

                        reponse = input("Retirer ou Ajouter? R/A:")
                        if reponse == 'R':
                            num = input('1 ou 2?')
                            if num == 1:
                                del(pasAR[compt])

                            else:
                                if num == 2:
                                    del(pasAR[compt + 1])
                        else:
                            if reponse == 'A':
                                deb = float(input('Debut du pas: ?'))

                                fin = float(input('Fin du pas: ?'))

                                newPas = [deb, -1, fin - deb + 1, 'ax', donnees._acc[0][deb:fin]]
                                pasAR.append(newPas)
                                pasAR.sort()

                                reponse2 = input(
                                    'Ajouter le nouveau pas à la bibliothèque? y/n :')
                                if reponse2 == 'y':
                                    #Chargement de la bibliothèque
                                    if donneesPied._capteur == 'Pied Gauche':
                                        bibliothequeString = 'listPatternsG'
                                        bibliotheque = Bibliotheque('listPatternsG.pickle')

                                    else:
                                        bibliothequeString = 'listPatternsD'
                                        bibliotheque = Bibliotheque('listPatternsD.pickle')

                                    miseAJour(donneesPied=donnees, bibliotheque=bibliotheque,
                                              bibliothequeString=bibliothequeString, newPas=newPas,
                                              filename=donnees._filename)

                                else:
                                    if reponse2 == 'n':
                                        pass
                else:
                    if probleme == 'n':
                        compt = compt + 1
            else:
                if version == 2.:
                    essai1, e0, e2 = parcours(donnees, donneesCeint, longFen=80,
                                              seuilBruit=1, seuilCor=0.72,
                                              posIni=pasAR[compt][0] + pasAR[compt][2],
                                              posFin=pasAR[compt+1][0])
                    if len(essai1) == 0:
                        compt = compt + 1
                    else:
                        #plt.ion()
                        #plt.figure()
                        #plt.plot(donnees._acc[0])
                        #plt.plot([pasAR[compt + 1][0], pasAR[compt + 1][0]], [-10, 10]
                        #          , 'k', linewidth=3.0)
                        #plt.plot([pasAR[compt][0] + pasAR[compt][2], pasAR[compt][0]
                        #         + pasAR[compt][2]], [-10, 10], 'r:', linewidth=3.0)
                        for k in range(len(essai1)):
                            #plt.plot([essai1[k][0], essai1[k][0]], [-10, 10], 'y', linewidth=3.0)
                            #plt.plot([essai1[k][0] + essai1[k][2], essai1[k][0] + essai1[k][2]],
                            #         [-10, 10], 'y:', linewidth=3.0)
                            pasAR.append(essai1[k])
                            pasAR.sort()
                            compteur = compteur + 1
        else:
            compt = compt + 1

    #print(compteur, 'pas récupérés grâce à testMediane')


def alternancePieds(listeD, listeG, donneesD, donneesG, donneesCeint):
    '''Verify if there is a right step detected between two left steps
       and the other way around. If not, 'parcours' is run on the signal
       where a step is supposed to be missing.

       Parameters:
       ----------
       listeD: step list (see parcours)
       listeG: step list (see parcours)

       donneesD: donnees object
                 right foot signal

       donneesG: donnees object
                 left foot signal

       donneesC: donnees object
                 waist signal
    '''

    if listeG[0][0] <= listeD[0][0]:
        a = [listeG[k][0] for k in range(len(listeG))]
        b = [listeD[k][0] for k in range(len(listeD))]
        u = 0

    else:
        a = [listeD[k][0] for k in range(len(listeD))]
        b = [listeG[k][0] for k in range(len(listeG))]
        u = 1

    compteurD = 0
    compteurG = 0

    while len(b) >= 1 and len(a) >= 2:
        if a[0] == b[0] or a[1] == b[0]:
            #print('Deux pas commmencent au meme endroit')
            pass
        elif a[0] < b[0] and a[1] > b[0]:
            a = a[1:]
            c = a
            a = b
            b = c
            u = (u+1) % 2
            pass
        else:
            if u == 0:
                essai1, e0, e2 = parcours(donneesD, donneesCeint, longFen=50,
                                          seuilBruit=1, seuilCor=0.67,
                                          posIni=a[0]+1,
                                          posFin=a[1]-1)
                if len(essai1) > 0:
                    for k in range(len(essai1)):
                        b.append(essai1[k][0])
                        compteurD = compteurD + 1
                        b.sort()
                        listeD.append(essai1[k])
                    listeD.sort()
                    a = a[1:]
                    c = a
                    a = b
                    b = c
                    u = u+1 % 2
                else:
                    #print("Risque d'oubli à droite entre ", a[0], " et ", a[1])
                    a = a[1:]

            else:
                essai1, e0, e2 = parcours(donneesG, donneesCeint, longFen=50,
                                          seuilBruit=1, seuilCor=0.67,
                                          posIni=a[0]+1,
                                          posFin=a[1]-1)
                if len(essai1) > 0:
                    for k in range(len(essai1)):
                            b.append(essai1[k][0])
                            compteurG = compteurG + 1
                            b.sort()
                            listeG.append(essai1[k])
                    listeG.sort()
                    a = a[1:]
                    c = a
                    a = b
                    b = c
                    u = u+1 % 2

                else:
                    #print("Risque d'oubli à gauche entre ", a[0], " et ", a[1])
                    a = a[1:]
    #print(compteurD, 'pieds droit rattrapés')
    #print(compteurG, 'pieds gauche rattrapés')


def RBM(sujet):
    '''Return two lists corresponding to the two feet segmentation, in which:
        -1st element: signal between O and the beginning of the 1st step
        -2nd element: list of the step vectors of the 1st way walking phase
        -3rd element: signal within the u-turn
        -4th element: list of the step vectors of the 2nd way walking phase
        -5th element: signal after the end of the last step

        Parameters:
        ----------

        sujet: SegStep object
              a SegStep object which has to be complete.
    '''
    sd = [[[], [], [], [], []] for i in range(6)]
    sg = [[[], [], [], [], []] for i in range(6)]
    finEnr = sujet.getTemps() - 1

    debAllD = sujet._pasAllerDroit[0][0]
    debAllG = sujet._pasAllerGauche[0][0]
    deb = min([debAllD, debAllG])

    debDmt = sujet._ceinture.getDemiTours()[0]
    finDmt = sujet._ceinture.getDemiTours()[1]

    finRetD = sujet._pasRetourDroit[-1][0]
    finRetG = sujet._pasRetourGauche[-1][0]

    fin = max([finRetD, finRetG])

    for i in range(3):
        sd[i][0] = sujet._pd._acc[i][0:deb]
        sd[i][1] = [sujet._pd._acc[i][sujet._pasAllerDroit[j][0]:
                    sujet._pasAllerDroit[j][0] + sujet._pasAllerDroit[j][2]]
                    for j in range(len(sujet._pasAllerDroit))]
        sd[i][2] = sujet._pd._acc[i][debDmt:finDmt]
        sd[i][3] = [sujet._pd._acc[i][sujet._pasRetourDroit[j][0]:
                    sujet._pasRetourDroit[j][0] + sujet._pasRetourDroit[j][2]]
                    for j in range(len(sujet._pasRetourDroit))]
        sd[i][4] = sujet._pd._acc[i][fin:finEnr]

    for i in range(3):
        sd[3 + i][0] = sujet._pd._gyr[i][0:deb]
        sd[3 + i][1] = [sujet._pd._gyr[i][sujet._pasAllerDroit[j][0]:
                                          sujet._pasAllerDroit[j][0] + sujet._pasAllerDroit[j][2]]
                        for j in range(len(sujet._pasAllerDroit))]
        sd[3 + i][2] = sujet._pd._gyr[i][debDmt:finDmt]
        sd[3 + i][3] = [sujet._pd._gyr[i][sujet._pasRetourDroit[j][0]:
                                          sujet._pasRetourDroit[j][0] + sujet._pasRetourDroit[j][2]]
                        for j in range(len(sujet._pasRetourDroit))]
        sd[3 + i][4] = sujet._pd._gyr[i][fin:finEnr]

    for i in range(3):
        sg[i][0] = sujet._pg._acc[i][0:deb]
        sg[i][1] = [sujet._pg._acc[i][sujet._pasAllerGauche[j][0]:
                                      sujet._pasAllerGauche[j][0] + sujet._pasAllerGauche[j][2]]
                    for j in range(len(sujet._pasAllerGauche))]
        sg[i][2] = sujet._pg._acc[i][debDmt:finDmt]
        sg[i][3] = [sujet._pg._acc[i][sujet._pasRetourGauche[j][0]:
                    sujet._pasRetourGauche[j][0] + sujet._pasRetourGauche[j][2]]
                    for j in range(len(sujet._pasRetourGauche))]
        sg[i][4] = sujet._pg._acc[i][fin:finEnr]

    for i in range(3):
        sg[3 + i][0] = sujet._pg._gyr[i][0:deb]
        sg[3 + i][1] = [sujet._pg._gyr[i][sujet._pasAllerGauche[j][0]:
                        sujet._pasAllerGauche[j][0] + sujet._pasAllerGauche[j][2]]
                        for j in range(len(sujet._pasAllerGauche))]
        sg[3 + i][2] = sujet._pg._gyr[i][debDmt:finDmt]
        sg[3 + i][3] = [sujet._pg._gyr[i][sujet._pasRetourGauche[j][0]:
                                          sujet._pasRetourGauche[j][0] +
                                          sujet._pasRetourGauche[j][2]]
                        for j in range(len(sujet._pasRetourGauche))]
        sg[3 + i][4] = sujet._pg._gyr[i][fin:finEnr]

    return sd, sg


def constructionListe(directory):
    import os

    result = []
    for filename in os.listdir(directory):
        a = filename.split('-')
        if len(a) < 4:
            print(filename)
        else:
            result.append([a[0]+'-'+a[1], a[3], a[2]])

    return result
