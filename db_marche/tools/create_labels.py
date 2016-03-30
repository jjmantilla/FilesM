import json
import logging
import os.path as osp
from csv import DictReader

from .. import DATA_DIR, Database
Label_Dir = osp.join(DATA_DIR, "Labels")


def parse_numerique(num):
    num = num.replace(',', '.')
    try:
        num = int(num)
    except ValueError:
        try:
            num = float(num)
        except ValueError:
            return -1
    return num


def parseline(row, row_type):
    res = {}
    for k, v in row.items():
        if row_type[k] == 'num':
            res[k] = parse_numerique(v)
        else:
            res[k] = v
    return res


def create_label_files(fname=osp.join(Label_Dir, 'Info_patient.csv'),
                       outdir=Label_Dir, l_id=None, debug=False):
    # Get the db ids (with labels)
    db = Database(debug=2 if debug else 1)
    ids = set([m['id'] for m in db.meta])

    # Get the exercices without labels
    exo_codes = l_id
    if not exo_codes:
        exo_codes = []
        import glob
        for pattern in [osp.join(DATA_DIR, 'Data', '*Tete.csv'),
                        osp.join(DATA_DIR, 'Raw', '*672.txt'),
                        osp.join(DATA_DIR, 'Raw', '*datas-1.txt')]:
            for fn in glob.glob(pattern):
                code = _process_name(fn)
                if code is not None:
                    exo_codes += [code]

    # Process the CSV file
    with open(fname) as f:
        csv_file = DictReader(f, delimiter=';')
        row_type = next(csv_file)
        new_files = []
        for row in csv_file:
            code = row['Code']
            id_exp = row['Numero enregistrement'].split()
            del row['Numero enregistrement']
            del row['Code']
            label = parseline(row, row_type)
            for i in id_exp:
                id_e = code+i
                if id_e in exo_codes:
                    if id_e not in ids:
                        new_files += ['{}_{}'.format(code, i)]
                    fname = code+'-'+i+'-label.js'
                    with open(osp.join(outdir, fname), 'w') as f:
                        json.dump(label, f)
        print('created {} new label file'.format(len(new_files)))
        ans = input('Show new? [y/N]')
        if len(ans) and ans[0].lower() == 'y':
            print(new_files)


def _process_name(fname):
    import re
    try:
        fname = path.basename(fname)
        g = re.search(r'^(\w{3}-\w{3,4})-YO-([0-9]+)-',
                      fname).groups()
        return g[0]+g[1]
    except AttributeError:
        logging.debug('No code in {}'.format(fname))
    return None
