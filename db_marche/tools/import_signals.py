import re
import os
import json
import shutil
import logging
from glob import glob
import os.path as osp
from .. import DATA_DIR, ID_SENSORS, Database
from ..utils import sane_update
from ..exercise import Exercise
from ..process.signal_loader import RejectedExerciseError

logger = logging.getLogger("Database")

Import_Dir = osp.join(DATA_DIR, "Import")
Label_Dir = osp.join(DATA_DIR, "Labels")
Raw_Dir = osp.join(DATA_DIR, "Raw")
Exo_Dir = osp.join(DATA_DIR, "Exo")
Labm_Dir = osp.join(DATA_DIR, "Manual_Annotation")


def import_signals(force=False, debug=1):
    db = Database(debug=debug)
    rejected_exo = []
    # Get all file possible for importation
    l_fname = glob(osp.join(Import_Dir, "*datas-1.txt"))
    l_fname += glob(osp.join(Import_Dir, "*672.txt"))
    print("errrt",l_fname)
    # Parse the ids of those files
    import_ids = [_get_id(fname) + (fname,) for fname in l_fname]
    existing_ids = [m['id'] for m in db.meta]
    rm_index = []
    for i, (c, e, _) in enumerate(import_ids):
        if c+e in existing_ids and not force:
            logger.warning("Exercise {} already exists in the Database. "
                           "It won't be added. If you are sure you need to "
                           "overwrite the existing one, use the --force "
                           "option.".format(c+e))
            rm_index += [i]
    rm_index.reverse()
    for i in rm_index:
        logger.debug(import_ids[i])
        del import_ids[i]
    print("errrt",import_ids)
    parse_labels = False
    for (code, n_exo, fname) in import_ids:
        id_ex = code + n_exo
        label_file = "{}-{}-label.js".format(code, n_exo)
        label_file = osp.join(Import_Dir, label_file)

        # If we cannot find the corresponding label file, try to parse
        # all the csv files in Import_Dir to create it
        if (not osp.exists(label_file) and not parse_labels):
            _find_labels(import_ids)
            parse_labels = True
        if not osp.exists(label_file):
            print(id_ex)
            rejected_exo += [(id_ex, RejectedExerciseError("No labels found"))]
            continue
        meta = {}
        import_file = fname.replace('000_00340672.txt', 'Xsens-PARTX.csv')
        import_file = import_file.replace('datas-1.txt', 'PARTX.csv')
        meta['fname'] = import_file
        meta['id'] = id_ex
        meta['sensor'] = 'TCon' if 'TCon' in fname else 'Xsens'
        with open(label_file) as f:
            lab = json.load(f)
            sane_update(meta, lab)
        try:
            ex = Exercise(meta, recompute='all')
        except RejectedExerciseError as e:
            rejected_exo += [(id_ex, e)]
            continue

        # Moves the files around
        fn = ex.fname.replace('Import', 'Exo')
        os.remove(ex._get_pkl_name())
        ex.fname = fn
        ex.meta['fname'] = fn
        ex._dump()
        assert osp.exists(ex._get_pkl_name())
        _move(label_file, Label_Dir)
        if meta['sensor'] == 'TCon':
            for i in range(1, 5):
                fmv = fname.replace('datas-1', 'datas-{}'.format(i))
                _move(fmv, Raw_Dir)
                fmv = fname.replace('datas-1', 'quats-{}'.format(i))
                _move(fmv, Raw_Dir)
        else:
            for sensor_id in ID_SENSORS:
                fmv = fname.replace('672.txt', '{}.txt'.format(sensor_id))
                _move(fmv, Raw_Dir)
        manual_seg = import_file.replace('-YO', '')
        manual_seg = manual_seg.replace('TCon-PARTX.csv', 'seg.labm')
        manual_seg = manual_seg.replace('Xsens-PARTX.csv', 'seg.labm')
        if osp.exists(manual_seg):
            _move(manual_seg, Labm_Dir)
            ex.load_manual_segmentation()

        manual_steps = import_file.replace('PARTX.csv', 'step.labm')
        if osp.exists(manual_steps):
            _move(manual_steps, Labm_Dir)
            ex.load_manual_steps()

    for id_ex, err in rejected_exo:
        print('{} - {}'.format(id_ex, err))


def _move(src, dest):
    fn = osp.join(dest, osp.basename(src))
    if osp.exists(fn):
        import os
        os.remove(fn)
    shutil.move(src, dest)


def _find_labels(l_id):
    from .create_labels import create_label_files
    csv_files = glob(osp.join(Import_Dir, "*csv"))
    l_id = [c+e for c, e, _ in l_id]
    for f in csv_files:
        create_label_files(fname=f, outdir=Import_Dir, l_id=l_id)


def _get_id(fname):
    fn = osp.basename(fname)
    g = re.search(r'([A-Z]{3}-\w{3,4})-YO-([0-9]+)-', fn)
    return g.groups()
