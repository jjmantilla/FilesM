import numpy as np
import scipy.signal


def feat_ForAvgSpeed(ex):
    '''Average speed during forward walking
    '''

    res = dict()
    seg = ex.seg_annotation
    distance = ex.meta.get('distance_parcourue_(m)')
    Fs = ex.meta.get('frequence')
    res['ForAvgSpeed'] = (distance * Fs / (seg[1] - seg[0]),
                          'Average speed during forward walking'
                          )
    return res


def feat_ForNorm(ex):
    '''Average and standard deviation of the norm of the acceleration during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        X_norm = np.sqrt(np.power(ex.data_sensor[k * 6, seg[0]:seg[1]], 2) + np.power(ex.data_sensor[
                         k * 6 + 1, seg[0]:seg[1]], 2) + np.power(ex.data_sensor[k * 6 + 2, seg[0]:seg[1]], 2))
        res['ForMeanNorm{}'.format(sensor)] = (
            X_norm.mean(),    # Velue of the feature
            'Mean of the norm of the acceleration for sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
        res['ForStdNorm{}'.format(sensor)] = (
            X_norm.std(),    # Velue of the feature
            'Standard deviation of the norm of the acceleration for sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res
#Xraw --> data_sensor

def feat_ForMeanAcc(ex):
    '''Mean of the acceleration for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            res['ForMeanAcc{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]].mean(),
                'Mean of the acceleration for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForMeanAccV{}'.format(sensor)] = (
            ex.data_earth[k * 6 + 2, seg[0]:seg[1]].mean(),    # Velue of the feature
            'Mean of the acceleration for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForMeanGyr(ex):
    '''Mean of the angular velocity for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(3, 'X'), (4, 'Y'), (5, 'Z')]:
            res['ForMeanGyr{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]].mean(),
                'Mean of the angular velocity for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForMeanGyrV{}'.format(sensor)] = (
            ex.data_earth[k * 6 + 5, seg[0]:seg[1]].mean(),    # Velue of the feature
            'Mean of the angular velocity for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForStdAcc(ex):
    '''Standard deviation of the acceleration for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            res['ForStdAcc{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]].std(),
                'Standard deviation of the acceleration for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForStdAccV{}'.format(sensor)] = (
            ex.data_earth[k * 6 + 2, seg[0]:seg[1]].std(),    # Velue of the feature
            'Standard deviation of the acceleration for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForStdGyr(ex):
    '''Standard deviation of the angular velocity for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(3, 'X'), (4, 'Y'), (5, 'Z')]:
            res['ForStdGyr{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]].std(),
                'Standard deviation of the angular velocity for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForStdGyrV{}'.format(sensor)] = (
            ex.data_earth[k * 6 + 5, seg[0]:seg[1]].std(),    # Velue of the feature
            'Standard deviation of the angular velocity for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForMaxAcc(ex):
    '''Maximum of the acceleration for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            res['ForMaxAcc{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]].max(),
                'Maximum of the acceleration for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForMaxAccV{}'.format(sensor)] = (
            ex.data_earth[k * 6 + 2, seg[0]:seg[1]].max(),    # Velue of the feature
            'Maximum of the acceleration for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForMaxGyr(ex):
    '''Maximum of the angular velocity for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(3, 'X'), (4, 'Y'), (5, 'Z')]:
            res['ForMaxGyr{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]].max(),
                'Maximum of the angular velocity for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForMaxGyrV{}'.format(sensor)] = (
            ex.data_earth[k * 6 + 5, seg[0]:seg[1]].max(),    # Velue of the feature
            'Maximum of the angular velocity for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForMinAcc(ex):
    '''Minimum of the acceleration for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            res['ForMinAcc{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]].min(),
                'Minimum of the acceleration for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForMinAccV{}'.format(sensor)] = (
            ex.data_earth[k * 6 + 2, seg[0]:seg[1]].min(),    # Velue of the feature
            'Minimum of the acceleration for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForMinGyr(ex):
    '''Minimum of the angular velocity for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(3, 'X'), (4, 'Y'), (5, 'Z')]:
            res['ForMinGyr{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]].min(),
                'Minimum of the angular velocity for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForMinGyrV{}'.format(sensor)] = (
            ex.data_earth[k * 6 + 5, seg[0]:seg[1]].min(),    # Velue of the feature
            'Minimum of the angular velocity for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForMedAcc(ex):
    '''Median of the acceleration for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            res['ForMedAcc{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                np.median(ex.data_sensor[k * 6 + i, seg[0]:seg[1]]),
                'Median of the acceleration for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForMedAccV{}'.format(sensor)] = (
            # Velue of the feature
            np.median(ex.data_earth[k * 6 + 2, seg[0]:seg[1]]),
            'Median of the acceleration for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForMedGyr(ex):
    '''Median of the angular velocity for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(3, 'X'), (4, 'Y'), (5, 'Z')]:
            res['ForMedGyr{}{}'.format(ax, sensor)] = (
                # Velue of the feature
                np.median(ex.data_sensor[k * 6 + i, seg[0]:seg[1]]),
                'Median of the angular velocity for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        res['ForMedGyrV{}'.format(sensor)] = (
            # Velue of the feature
            np.median(ex.data_earth[k * 6 + 5, seg[0]:seg[1]]),
            'Median of the angular velocity for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def _compute_DSP(x, Fs=100):
    nWindow = np.power(2, np.floor(np.log2(len(x) - 1)))
    nfft = 2048
    x = x - np.mean(x)
    w = scipy.signal.get_window("hamming", nWindow)
    fDSP, tDSP, DSP = scipy.signal.spectrogram(
        x, Fs, w, nWindow, nWindow - 1, nfft, False, True, 'density')
    ma_DSP = np.mean(DSP, axis=1)
    ma_DSP = np.divide(ma_DSP,sum(ma_DSP))
    return (fDSP, ma_DSP, nfft)


def _parabolic(f, x):
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)


def _compute_f0(DSP, nfft, Fs=100):
    i = np.argmax(DSP)
    true_i = _parabolic(np.log(DSP), i)[0]
    return Fs * i / nfft


def feat_ForFreqAcc(ex):
    '''Frequency information of the acceleration for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            fDSP, DSP, nfft = _compute_DSP(
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]], ex.meta.get('frequence'))
            f0 = _compute_f0(DSP, nfft, ex.meta.get('frequence'))
            Eb1 = sum(DSP[np.where((fDSP > 0) & (fDSP < 3))])
            Eb2 = sum(DSP[np.where((fDSP >= 3) & (fDSP < 7))])
            Eb3 = sum(DSP[np.where((fDSP >= 7) & (fDSP < 20))])
            res['ForF0Acc{}{}'.format(ax, sensor)] = (
                f0,    # Velue of the feature
                'Fundamental frequency of the acceleration for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
            res['ForE03Acc{}{}'.format(ax, sensor)] = (
                Eb1,    # Velue of the feature
                'Energy of the acceleration in frequency band 0-3 Hz for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
            res['ForE37Acc{}{}'.format(ax, sensor)] = (
                Eb2,    # Velue of the feature
                'Energy of the acceleration in frequency band 3-7 Hz for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
            res['ForE720Acc{}{}'.format(ax, sensor)] = (
                Eb3,    # Velue of the feature
                'Energy of the acceleration in frequency band 7-20 Hz for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        fDSP, DSP, nfft = _compute_DSP(
            ex.data_earth[k * 6 + 2, seg[0]:seg[1]], ex.meta.get('frequence'))
        f0 = _compute_f0(DSP, nfft, ex.meta.get('frequence'))
        Eb1 = sum(DSP[np.where((fDSP > 0) & (fDSP < 3))])
        Eb2 = sum(DSP[np.where((fDSP >= 3) & (fDSP < 7))])
        Eb3 = sum(DSP[np.where((fDSP >= 7) & (fDSP < 20))])
        res['ForF0AccV{}'.format(sensor)] = (
            f0,    # Velue of the feature
            'Fundamental frequency of the acceleration for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
        res['ForE03AccV{}'.format(sensor)] = (
            Eb1,    # Velue of the feature
            'Energy of the acceleration in frequency band 0-3 Hz for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
        res['ForE37AccV{}'.format(sensor)] = (
            Eb2,    # Velue of the feature
            'Energy of the acceleration in frequency band 3-7 Hz for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
        res['ForE720AccV{}'.format(sensor)] = (
            Eb3,    # Velue of the feature
            'Energy of the acceleration in frequency band 7-20 Hz for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res


def feat_ForFreqGyr(ex):
    '''Frequency information of the angular velocity for each axe during forward walking
    '''
    res = dict()
    seg = ex.seg_annotation
    for k, sensor in enumerate(['RightFoot', 'LeftFoot',
                                'Waist', 'Head']):
        for i, ax in [(3, 'X'), (4, 'Y'), (5, 'Z')]:
            fDSP, DSP, nfft = _compute_DSP(
                ex.data_sensor[k * 6 + i, seg[0]:seg[1]], ex.meta.get('frequence'))
            f0 = _compute_f0(DSP, nfft, ex.meta.get('frequence'))
            Eb1 = sum(DSP[np.where((fDSP > 0) & (fDSP < 3))])
            Eb2 = sum(DSP[np.where((fDSP >= 3) & (fDSP < 7))])
            Eb3 = sum(DSP[np.where((fDSP >= 7) & (fDSP < 20))])
            res['ForF0Gyr{}{}'.format(ax, sensor)] = (
                f0,    # Velue of the feature
                'Fundamental frequency of the angular velocity for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
            res['ForE03Gyr{}{}'.format(ax, sensor)] = (
                Eb1,    # Velue of the feature
                'Energy of the angular velocity in frequency band 0-3 Hz for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
            res['ForE37Gyr{}{}'.format(ax, sensor)] = (
                Eb2,    # Velue of the feature
                'Energy of the angular velocity in frequency band 3-7 Hz for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
            res['ForE720Gyr{}{}'.format(ax, sensor)] = (
                Eb3,    # Velue of the feature
                'Energy of the angular velocity in frequency band 7-20 Hz for axe {} and sensor {} during forward walking'
                ''.format(ax, sensor)  # Doc string
            )
        fDSP, DSP, nfft = _compute_DSP(
            ex.data_earth[k * 6 + 5, seg[0]:seg[1]], ex.meta.get('frequence'))
        f0 = _compute_f0(DSP, nfft, ex.meta.get('frequence'))
        Eb1 = sum(DSP[np.where((fDSP > 0) & (fDSP < 3))])
        Eb2 = sum(DSP[np.where((fDSP >= 3) & (fDSP < 7))])
        Eb3 = sum(DSP[np.where((fDSP >= 7) & (fDSP < 20))])
        res['ForF0GyrV{}'.format(sensor)] = (
            f0,    # Velue of the feature
            'Fundamental frequency of the angular velocity for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
        res['ForE03GyrV{}'.format(sensor)] = (
            Eb1,    # Velue of the feature
            'Energy of the angular velocity in frequency band 0-3 Hz for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
        res['ForE37GyrV{}'.format(sensor)] = (
            Eb2,    # Velue of the feature
            'Energy of the angular velocity in frequency band 3-7 Hz for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
        res['ForE720GyrV{}'.format(sensor)] = (
            Eb3,    # Velue of the feature
            'Energy of the angular velocity in frequency band 7-20 Hz for axe V and sensor {} during forward walking'
            ''.format(sensor)  # Doc string
        )
    return res
