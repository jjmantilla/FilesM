# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from db_marche.process.functionSeg import *
import statistics
from sys import stdout as out


class SegStep:
    """Class containing the step detection of an acquisition

        Usage:
        -----
        Construct a object with the 4 captors signals of an experience
        To get a normalized signal (independent from the gravity) use zScale()
        To get the steps, use getPas()
        To get the segmentation figure, use getGraph()
        To get the three previous actions at once, use getAll()

    """

    def __init__(self, X, seg, freq):
        '''Return a SegStep object filled with the signals, the sample frequence,
           and the u-turn detection.
           This constructor does NOT process the segmentation.

           Parameters:
           ----------
            X: numpy.array
                Matrix with the 24 signals corresponding to one exercise (see Exercise.get_signal())
            seg: list of 4 elements
                - beginning of the first walking phase
                - beginning of the u-turn
                - end of the u-turn
                - end of the second walking phase
            freq: float
                Sampling frequency
        '''
        #Demi-Tour:
        self.dmt = seg

        #Données pied droit
        self._pd = donnees(x0=X[0], x1=X[1], x2=X[2], x3=X[3], x4=X[4], x5=X[5], dmt=seg)
        self._pd._capteur = 'Pied Droit'

        #Données pied gauche
        self._pg = donnees(x0=X[6], x1=X[7], x2=X[8], x3=X[9], x4=X[10], x5=X[11], dmt=seg)
        self._pg._capteur = 'Pied Gauche'

        #Données ceinture
        self._ceinture = donnees(x0=X[12], x1=X[13], x2=X[14], x3=X[15],
                                 x4=X[16], x5=X[17], dmt=seg)
        self._ceinture._capteur = 'Ceinture'

        #Données tete
        self._tete = donnees(x0=X[18], x1=X[19], x2=X[20], x3=X[21],
                             x4=X[22], x5=X[23], dmt=seg)
        self._tete._capteur = 'Tete'

        #Frequence
        self.freq = freq

        #Initialisation des pas
        self._pasAllerDroit = []
        self._pasRetourDroit = []

        self._pasAllerGauche = []
        self._pasRetourGauche = []

        self._pasGauche = []
        self._pasDroit = []

    def zScale(self):
        '''Normalize all the donnees objects of the the SegStep object (see donnees.zScale())
        '''

        self._pg.zScale()
        self._pd.zScale()
        self._ceinture.zScale()
        self._tete.zScale()

    def getPas(self, version, incr=False):
        '''Process the segmentation of the signals in the SegStep object.

           Parameters:
           ----------
           version= 1. or 2.
                 1: The algorithm is interactive, if a problem is detected via
                    functionSeg.testAmp or functionSeg.testMediane, then the user
                    has to process manually a part of the segmentation.
                    This version and only this version enable the user to fulfill
                    the pattern library.

                 2: The algorithm is automatic (see functionSeg.testAmp
                    & functionSeg.testMediane for details)

            incr = bool, optional (default: False)
                 select if the segmentation has to uptade the pattern score

        '''
        from .functionSeg import parcours, testAmp, testMediane

        out.write('\b'*40)
        out.write('\tSegmentation du pied droit'.ljust(40))
        out.flush()
        (pasD, pasAllerD, pasRetourD) = parcours(self._pd, self._ceinture, incr=incr)
        testAmp(self._pd, pasAllerD, version=version)
        testAmp(self._pd, pasRetourD, version=version)
        testMediane(self._pd, self._ceinture, pasAllerD, version=version)
        testMediane(self._pd, self._ceinture, pasRetourD, version=version)

        out.write('\b'*40)
        out.write('\tSegmentation du pied gauche'.ljust(40))
        out.flush()
        (pasG, pasAllerG, pasRetourG) = parcours(self._pg, self._ceinture, incr=incr)
        testAmp(self._pg, pasAllerG, version=version)
        testAmp(self._pg, pasRetourG, version=version)
        testMediane(self._pg, self._ceinture, pasAllerG, version=version)
        testMediane(self._pg, self._ceinture, pasRetourG, version=version)

        self._pasAllerGauche = pasAllerG
        self._pasRetourGauche = pasRetourG

        self._pasAllerDroit = pasAllerD
        self._pasRetourDroit = pasRetourD

        out.write('\b'*40)
        out.write('\tTest pied droit--pied gauche'.ljust(40))
        out.flush()
        self._pasGauche = self._pasAllerGauche + self._pasRetourGauche
        self._pasDroit = self._pasAllerDroit + self._pasRetourDroit

        out.write('\b'*40)
        out.write('\tTest en amplitude'.ljust(40))
        out.flush()
        testAmp(self._pd, pasAllerD, version=version)
        testAmp(self._pd, pasRetourD, version=version)
        testAmp(self._pg, pasAllerG, version=version)
        testAmp(self._pg, pasRetourG, version=version)

    def getPasCalibres(self):
        ''' Function to train a neural network calibrating the steps
        '''
        test = librairiePasCalibres.load()
        for pas in self._pasDroit[0:5]:

            vect = (self._ceinture._acc[2][pas[0]:pas[0]+pas[2]])
            plt.ion()
            plt.figure()
            plt.plot(vect)
            plt.plot(pas[4])
            plt.show()

            deb = int(input('début du vrai pas? : '))
            fin = int(input('fin du vrai pas? : '))

            save = input('Sauvegarder? y/n : ')

            if save == 'y':
                test.append(pasCalibre(vectR=vect, deb=deb, fin=fin))
                test.save()
            else:
                pass

        for pas in self._pasGauche[0:5]:

            vect = (self._ceinture._acc[2][pas[0]:pas[0]+pas[2]])
            plt.ion()
            plt.figure()
            plt.plot(vect)
            plt.plot(pas[4])
            plt.show()

            deb = int(input('début du vrai pas? : '))
            fin = int(input('fin du vrai pas? : '))

            save = input('Sauvegarder? y/n : ')

            if save == 'y':
                test.append(pasCalibre(vectR=vect, deb=deb, fin=fin))
                test.save()
            else:
                pass

    def getGraph(self):
        '''Draw final segmentation of the right foot signal and the left foot signal
        '''

        plt.ioff()
        plt.figure('Seg_step')
        plt.subplot(211)
        #plt.plot(self._ceinture._acc[2])
        plt.plot(self._pd._acc[0])

        for i in range(len(self._pasDroit)):
            plt.plot([self._pasDroit[i][0], self._pasDroit[i][0]], [
                     -10, 10], 'g', linewidth=2.0)

            plt.text(self._pasDroit[i][0], 11, str(self._pasDroit[i][1]))
            plt.plot([self._pasDroit[i][0] + self._pasDroit[i][2], self._pasDroit[
                     i][0] + self._pasDroit[i][2]], [-10, 10], 'm:', linewidth=2.0)

        plt.legend(['Acceleration axe antero posterieur', 'Debut de pas', 'Fin de pas'],
                   loc='upper left', prop={'size': 8})

        for i in range(len(self._pasDroit)):
            plt.plot([self._pasDroit[i][0] + self._pasDroit[i][2], self._pasDroit[
                     i][0] + self._pasDroit[i][2]], [-10, 10], 'm:', linewidth=2.0)
        plt.plot([self.dmt[0]]*2, [-10, 10], linewidth=3.0, c='c')
        plt.plot([self.dmt[1]]*2, [-10, 10], linewidth=3.0, c='c')

        plt.xlabel('Temps 1:100 sec')
        plt.ylabel('Acceleration m^2.sec^-1')

        plt.title('Segmentation Pied Droit - Pied Gauche')

        plt.subplot(212)
        #plt.plot(self._ceinture._acc[2])
        plt.plot(self._pg._acc[0])

        for i in range(len(self._pasGauche)):
            plt.plot([self._pasGauche[i][0], self._pasGauche[i][0]],
                     [-10, 10], 'g', linewidth=2.0)
            plt.text(self._pasGauche[i][0], 11, str(self._pasGauche[i][1]))
            plt.plot([self._pasGauche[i][0] + self._pasGauche[i][2], self._pasGauche[
                     i][0] + self._pasGauche[i][2]], [-10, 10], 'm:', linewidth=2.0)
        plt.legend(
            ['Acceleration axe antero posterieur', 'Debut de pas',
             'Fin de pas'], loc='upper left', prop={'size': 8})

        for i in range(len(self._pasGauche)):
            plt.plot([self._pasGauche[i][0] + self._pasGauche[i][2], self._pasGauche[
                     i][0] + self._pasGauche[i][2]], [-10, 10], 'm:', linewidth=2.0)

        plt.plot([self.dmt[0]]*2, [-10, 10], linewidth=3.0, c='c')
        plt.plot([self.dmt[1]]*2, [-10, 10], linewidth=3.0, c='c')

        plt.xlabel('Temps 1:100 sec')
        plt.ylabel('Acceleration m^2.sec^-1')
        plt.draw()
        plt.show()

    def getTemps(self):
        '''Return the number of points in an acquisition
        '''
        return len(self._pd._acc[0])

    def getAll(self, version=2, graph=True):
        '''Build the SegStep object with the step detection through the methods
           zScale(), getPas(), getGraph()

           If the frequency sampling is different from 100Hz, a message is print
        '''
        if self.freq != 100:
            print("la fréquence n'est pas correcte")
        else:
            self.zScale()

            self.getPas(version)
            if graph:
                self.getGraph()
            #self.getPasCalibres()

    def compute_steps(self, version=2, graph=True):
        ''' Return a list of 4 lists of a couple of floats
                The 4 elements correspond to the two walking-phases with each step.
                    The first element of the couple is the beginning of a step.
                    The second elemnent of the couple is the end of a step.
            Parameters:
            ----------
            version= 1. or 2.
                Stands for the version of the algorithm (see getPas() for details)
            graph
                If True, display the segmentation once it is done
        '''
        self.getAll(version, graph=graph)
        steps = []
        steps.append(self._pasAllerDroit)
        steps.append(self._pasRetourDroit)
        steps.append(self._pasAllerGauche)
        steps.append(self._pasRetourGauche)
        res = []
        for ls, seg in zip(steps, ['Aller_D', 'Retour_D', 'Aller_G', 'Retour_G']):
            res.append([])
            for s in ls:
                res[-1].append((s[0], s[0]+s[2], s[1], s[5]))
        return res


class donnees:
    """Class containing the signals of one captor

        Usage:
        -----
        The object is built from a part of the data matrix X, and the walking phase segmentation
        To normalize the signals use zScale()
        To get the u-turn, use getDemiTours()
        To get the number of points of an acquisition, use getTemps()

    """
    def __init__(self, x0, x1, x2, x3, x4, x5, dmt):
        ''' Build a donnees object from the 6 vector signals of a captor and the
            walking phase segmentation.

            Parameters:
            ----------

            x0, x1, x2: numpy.array
                corresponding to the 3 acceleration signals
            x3, x4, x5: numpy.array
                corresponding to the 3 angular velocity signals
            dmt: list of 4 elements:
                - beginning of the first walking phase
                - beginning of the u-turn
                - end of the u-turn
                - end of the second walking phase
        '''
        self._acc = [x0, x1, x2]
        self._gyr = [x3, x4, x5]

        self._acc[0] = np.array(self._acc[0])
        self._acc[1] = np.array(self._acc[1])
        self._acc[2] = np.array(self._acc[2])

        self._gyr[0] = np.array(self._gyr[0])
        self._gyr[1] = np.array(self._gyr[1])
        self._gyr[2] = np.array(self._gyr[2])

        self._dmt = dmt

    def getDemiTours(self):
        '''Return the walking phase segmentation
        '''
        return self._dmt

    def zScale(self):
        ''' Normalize the vertical acceleration signal.

        Usage:
        -----
        To find the component of the acceleration due to gravity, the method uses the median
        of the vertical acceleration in the 3 first second where the patient is suposed to be
        immobile.
        '''
        m = np.median(self._acc[2][0:300])
        self._acc[2] = self._acc[2] - m

    def getTemps(self):
        return len(self._acc[0])

    def getNoise(self, n=5, l=500):
        ''' Return the noise standard deviation of a window sliding
            on the first part of the signal.

            Parameters:
            ----------
            n : int (default = 5)
                size of the window

            l : int (default = 300)
                size of the first part of the signal to ake in account

        '''
        from .functionSeg import testBruit2

        l = self.getTemps()

        BruitAZ = []
        for i in range(l-n-1):
            BruitAZ.append(testBruit2(self._acc[2][i:i+n-1]))

        m = statistics.mean(BruitAZ)
        std = statistics.stdev(BruitAZ)

        return m, std
