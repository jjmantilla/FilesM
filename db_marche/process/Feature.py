import numpy as np
import scipy.signal
# Hard coded!!
# TODO: change to adapt to the library
LEN_BIB = 134


class Feature(object):
    """Feature computation"""
    def __init__(self):
        pass

    def compute(self, exo):
        feats = []
        feat_names = []
        acc = []
        for pied in 'DG':
            for axe in 'XYZ':
                acc.append(exo.get_signal(desc=pied+'A'+axe)[0])#, seg='AR'))
        acc = np.array(acc)
        acc -= acc.mean(axis=1).reshape((-1, 1))
        norm_accD = np.sqrt((acc[:3]**2).sum(axis=0))
        norm_accG = np.sqrt((acc[3:]**2).sum(axis=0))

        M1 = norm_accD.mean()
        M2 = norm_accG.mean()
        Std1 = norm_accD.std()
        Std2 = norm_accG.std()
        feat_names += ['M_diffGD_norm_acc']
        feat_names += ['S_diffGD_norm_acc']
        feats += [abs(M1-M2), abs(Std1-Std2)]

        for k in 'DGCT':
            for ax in 'XYZ':
                acc = exo.get_signal(k+'A'+axe)
                fa = self.filter(self,acc, exo.fps)
                vit = np.array([scipy.integrate.simps(fa[:t], dx=exo.fps)
                                for t in range(1, len(fa))])
                pos = np.array([scipy.integrate.simps(vit[:t], dx=exo.fps)
                                for t in range(1, len(fa))])
                feat_names += ['Amp_Acc_'+k+'_'+ax]
                feats += [max(fa) - min(fa)]
                for ref, x in [('Acc', acc), ('Speed', vit), ('Pos', pos)]:
                    for n_seg, seg in zip(['Forth', 'Back'], range(2)):
                        x_seg = x[exo.seg[seg*2]:exo.seg[2*seg+1]]
                        [MM, NN] = self.FFT(x_seg)
                        N1 = [0, 0, 0, 0]
                        for fr in range(len(MM)):
                            if MM[fr] < 1:
                                N1[0] += NN[fr]
                            elif MM[fr] < 3:
                                N1[1] += NN[fr]
                            elif MM[fr] < 10:
                                N1[2] += NN[fr]
                            else:
                                N1[3] += NN[fr]
                        SS = N1[0] + N1[1] + N1[2] + N1[3]
                        feats += [N1[0] / SS]
                        feat_names += ['FFT 0-1Hz_'+ref+'_'+ax+'_' +
                                       n_seg+'_'+k+'_']
                        feats += [N1[1] / SS]
                        feat_names += ['FFT 1-3Hz_'+ref+'_'+ax+'_' +
                                       n_seg+'_'+k+'_']
                        feats += [N1[2] / SS]
                        feat_names += ['FFT 3-10Hz_'+ref+'_'+ax+'_' +
                                       n_seg+'_'+k+'_']
                        feats += [N1[3] / SS]
                        feat_names += ['FFT 10+Hz_'+ref+'_'+ax+'_' +
                                       n_seg+'_'+k+'_']
                        feat_names += ['Amp_'+ref+'_'+k+'_'+ax+'_'+n_seg]
                        feats += [max(x_seg) - min(x_seg)]
                        feat_names += ['Mean_'+ref+'_'+k+'_'+ax+'_'+n_seg]
                        feats += [x_seg.mean()]
                        feat_names += ['Std_'+ref+'_'+k+'_'+ax+'_'+n_seg]
                        feats += [x_seg.std()]
        if len(exo.steps) > 0:
            sf, sfn = self.steps_features(exo.steps, exo.fps)
            feats += sf
            feat_names += sfn
        return feats, feat_names

    def filter(self, signal, samplingFreq):
        Fs = samplingFreq
        f_min = 0.5
        [b, a] = scipy.signal.butter(2, [f_min / (Fs / 2)], 'high')
        x = scipy.signal.filtfilt(b, a, signal)
        return(x)

    def FFT(self, sig):
        n = len(sig)  # length of the signal
        k = np.arange(n)
        T = n / 100
        frq = k / T  # two sides frequency range
        frq = frq[:int(n / 2)]  # one side frequency range
        Y = scipy.fft(sig) / n  # fft computing and normalization
        Y = Y[:int(n / 2)]
        return([frq, abs(Y)])  # returning the spectrum

    def steps_features(self, steps, fps):
        f_name = []
        feat = []
        seg_names = ['aller_D', 'retour_D', 'aller_G', 'retour_G']
        for seg, name in zip(steps, seg_names):
            d_bib = np.array([s[3] for s in seg])
            homo = set([s[2] for s in seg])
            seg = np.array([[s[0], s[1]] for s in seg])
            step_length = [(seg[:, 1] - seg[:, 0])/fps]
            inter_step = [(seg[1:, 0] - seg[:-1, 1])/fps]
            f_name += ['mean_len-step_'+name]
            feat += [np.mean(step_length)]
            f_name += ['std_len-step_'+name]
            feat += [np.std(step_length)]
            f_name += ['amp_len-step_'+name]
            feat += [np.max(step_length)-np.min(step_length)]
            f_name += ['mean_inter-step_'+name]
            feat += [np.mean(inter_step)]
            f_name += ['std_inter-step_'+name]
            feat += [np.std(inter_step)]
            f_name += ['amp_inter-step_'+name]
            feat += [np.max(inter_step)-np.min(inter_step)]
            f_name.append('Homogeneity_step_'+name)
            feat.append(len(homo)/LEN_BIB)
            f_name.append('M_dist_bib_'+name)
            feat.append(d_bib.mean())
            f_name.append('S_dist_bib_'+name)
            feat.append(d_bib.std())

        l_aller = max(steps[2][-1][1], steps[0][-1][1]) - min(steps[2][0][0], steps[0][0][0])
        l_retour = max(steps[3][-1][1], steps[1][-1][1]) - min(steps[3][0][0], steps[1][0][0])

        aller_da = np.zeros(l_aller, dtype=int)
        for s in steps[0]:
            aller_da[s[0]:s[1]] = 1
        for s in steps[2]:
            aller_da[s[0]:s[1]] &= 1

        retour_da = np.zeros(l_retour, dtype=int)
        for s in steps[1]:
            retour_da[s[0]:s[1]] = 1
        for s in steps[3]:
            retour_da[s[0]:s[1]] &= 1

        t_double_appui = (sum(aller_da) + sum(retour_da))/fps
        t_double_appui_aller = sum(aller_da)/fps
        t_double_appui_retour = sum(retour_da)/fps

        f_name += ['T_double_appui']
        feat += [t_double_appui]
        f_name += ['T_double_appui_aller']
        feat += [t_double_appui_aller]
        f_name += ['T_double_appui_retour']
        feat += [t_double_appui_retour]
        f_name += ['D_AR_T_double_appui']
        feat += [t_double_appui_aller-t_double_appui_retour]
        f_name += ['l_retour']
        feat += [l_retour]
        f_name += ['l_aller']
        feat += [l_aller]

        return feat, f_name
