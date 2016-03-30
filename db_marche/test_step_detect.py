import numpy as np
from db_marche.process.step_detection import StepDetection
from db_marche import Database
from db_marche.process.pattern import Pattern
import  matplotlib.pyplot as plt
import pylab


def test_stepdetect(db):
    #db = Database()
    l_ex = db.get_data(limit=2)#, atteinte='Temoin')
    for k, ex in enumerate(l_ex):
        if ex.steps_annotation is None:
            continue

        # Build a Pattern library with the steps
        patterns = []
        for foot in range(2):
            for st in ex.steps_annotation[foot]:
                if st[1]-st[0] < 30:
                    continue
                patterns += [Pattern(dict(coord='RY', l_pat=st[1]-st[0],
                                          foot='right' if foot else 'left'),
                                     ex.data_sensor[6*foot+4, st[0]:st[1]])]
                patterns += [Pattern(dict(coord='AZ', l_pat=st[1]-st[0],
                                          foot='right' if foot else 'left'),
                                     ex.data_sensor[6*foot+2, st[0]:st[1]])]
                patterns += [Pattern(dict(coord='AV', l_pat=st[1]-st[0],
                                          foot='right' if foot else 'left'),
                                     ex.data_earth[6*foot+2, st[0]:st[1]])]
        stepDet = StepDetection(patterns=patterns, lmbd=.8, mu=.1)






        # Perform the step detection
        steps, steps_label = stepDet.compute_steps(ex)
        return steps,steps_label
        # plt.subplot(131);plt.plot(ex.DAY.T[steps[0][0][0]-10:steps[0][0][1]+10])
        # plt.subplot(132);plt.plot(ex.DAX.T[steps[0][0][0]-10:steps[0][0][1]+10])
        # plt.subplot(133);plt.plot(ex.DAZ.T[steps[0][0][0]-10:steps[0][0][1]+10])
        # seg = ex.seg_annotation
        # pylab.show()
        # plt.subplot(131); plt.plot(ex.DAZ.T[seg[0]:seg[1]])  # Forward walk
        # plt.subplot(132); plt.plot(ex.DAZ.T[seg[1]:seg[2]])
        # plt.subplot(133); plt.plot(ex.DAZ.T[seg[2]:seg[3]])
        #
        # pylab.show()


        # num_pat = [l//3 for f in range(2) for l in steps_label[f]]
        # output = []
        # for foot in range(2):
        #     for l in steps_label[foot]:
        #         pat = stepDet.patterns[l]
        #         output += ["{}: {} - {}".format(
        #             l//3+1, pat.meta['coord'], pat.meta['foot'])]
        # output = "\n".join(output)
        # label = np.arange(len(steps[0])+len(steps[1]))
        # i, failure = 0, []
        # for j, s in enumerate(num_pat):
        #     if s == label[i]:
        #         i += 1
        #     else:
        #         pat = stepDet.patterns[s]
        #         failure += [(j, s, pat.meta)]
        # assert np.allclose(num_pat, label), (
        #     "{}\n\n{} : {}".format(
        #         output, k, failure))
#test_stepdetect()