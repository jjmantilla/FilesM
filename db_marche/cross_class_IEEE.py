from db_marche import Database
from db_marche.tools.mk_pattern_library import mk_rand_pattern_library
from db_marche.process.step_detection import StepDetection

db = Database(debug=-1)
N = db.n_samples

PRs = []
from joblib import Parallel, delayed
from db_marche.metrics import count_hit
import numpy as np
from time import time

results = []
n_batch = 400

P = 10
lmbd = .8
mu = .1
CLASS_A = {'impressionclinicien': 0}
CLASS_B = {'impressionclinicien': '>2'}
CLASS = [('Sain', CLASS_A, [0]),
         ('Patho', CLASS_B, [3, 4])]

CLASS_train = CLASS + [(
    'Sain+Patho',
    {'weigths': [(.5, CLASS_A), (.5, CLASS_B)]},
    None)]


simu = 0
for c_train in CLASS_train:
    for c_test in CLASS:
        simu += 1
        res = dict(lmbd=.8, mu=.1, P=10, results=[],
                   c_train=c_train[0], c_test=c_test[0],
                   patterns=[])
        t_start = time()
        for j in range(100):
            patterns, codes = mk_rand_pattern_library(
                P=10, db=db, **c_train[1])
            sd = StepDetection(lmbd=lmbd, mu=mu, patterns=patterns)

            def PR(p):
                ex = db.load_data_object(p)
                if ex.impressionclinicien not in c_test[2]:
                    return None
                steps, _ = sd.compute_steps(ex)
                ch_p, ch_r, lytf, lypf = 0, 0, 0, 0
                for ytf, ypf in zip(ex.steps_annotation, steps):
                    ch_p += count_hit(ytf, ypf)
                    ch_r += count_hit(ypf, ytf)
                    lytf += len(ytf)
                    lypf += len(ypf)
                if lytf == 0:
                    print('Warning: No steps annotation for', ex.id)
                    ch_r, lytf = 0, 1
                if lypf == 0:
                    ch_p, lypf = 0, 1
                return [ch_r/lytf, ch_p/lypf]

            del_PR = delayed(PR)

            PRS = Parallel(n_jobs=-2)(del_PR(ex) for ex in range(N))
            PRS = np.array([prs for prs in PRS if prs is not None])
            t_batch = time()-t_start
            score = list(PRS.mean(axis=0))
            score += list(PRS.std(axis=0))

            res['results'] += [score]
            res['patterns'] += [patterns]
        results += [res]
        t_batch = time() - t_start
        print('-'*79)
        print('Batch : {:03}/{:03}'.format(simu, n_batch))
        print('Time batch : {:.2f}s'.format(t_batch))

        print('Train: {}, Test: {}'.format(c_train[0], c_test[0]))
        print('Score: {0:.2f}({2:.2f}), {1:.2f} ({3:.2f})'
              ''.format(*(np.mean(res['results'], axis=0))))
        print('-'*79+'\n\n')
np.save('save_simu/cross_class_IEEE.npy', results)
