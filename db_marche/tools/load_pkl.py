from sys import stdout as out

from db_marche import Database


# def reload_pkl(recompute=''):
#     db = Database()

#     for j, m in enumerate(db.meta):
#         out.write('\rReload db: {:7.2%} '.format(j/len(db.meta)))
#         out.flush()
#         try:
#             Exercise(meta=m, pkl_reload=True,
#                      recompute=recompute)
#         except Exception:
#             import traceback
#             msg = traceback.format_exc()
#             print('\n'+'-'*79)
#             print('Fail to load {} with traceback'.format(m['id']))
#             print('-'*79)
#             print(msg)
#             print('-'*79)
#     print('\rReload db: {:7}'.format('done'))


def recompute_ex(ex, step, seg, j, N):
    out.write('\rRecompute db: {:7.2%} '.format(j/N))
    out.flush()
    if step:
        ex.load_steps(recompute=True)
    if seg:
        ex.load_segmentation(recompute=True)


def recompute_db(selector, debug=False):
    dbg = 2 if debug else 1
    seg = 'seg' in selector
    step = 'step' in selector

    db = Database(debug=dbg)
    lex = db.get_data()
    from joblib import delayed, Parallel
    Parallel(n_jobs=-1, verbose=5)(
        delayed(recompute_ex)(ex, step, seg, j, len(lex))
        for j, ex in enumerate(lex))
    print('\rRecompute db: {:7}'.format('done'))


def reload_exo(meta, pkl_reload, recompute, j, N):
    from db_marche.exercise import Exercise
    Exercise(meta, pkl_reload=pkl_reload,
             recompute=recompute)
    from sys import stdout as out
    out.write('\rReload: {:7.2%}'.format(j/N))
    out.flush()


def reload_db(pkl_reload=False, recompute='', debug=False):
    dbg = 2 if debug else 1
    db = Database(debug=dbg)
    #from joblib import Parallel, delayed

    #N = len(db.meta)
    #Parallel(n_jobs=-1)(
    #    delayed(reload_exo)(m, pkl_reload, recompute, j, N)
    #    for j, m in enumerate(db.meta))
    #return
    db.get_data(load_args=dict(pkl_reload=pkl_reload,
                               recompute=recompute))
