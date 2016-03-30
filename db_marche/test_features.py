from db_marche import Database
from db_marche._import_features import compute_features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Test the feature loading')
    parser.add_argument('-n', type=int, default=1,
                        help='# of exercices to test')
    parser.add_argument('--code', type=str, default=None,
                        help='Specify an exercise to runn on')
    args = parser.parse_args()

    db = Database()
    list_exercice = db.get_data(limit=args.n, code=args.code)

    feats = []
    feat = None
    for ex in list_exercice:
        feat = compute_features(ex=ex)
        feats += [feat]

    # print('Finish with feat = ', feat)
    print('Total with ', feats[0][0], ' features')
    print(feats)
    # for key, values in sorted(feats[0][1].items()):
    #     print(key,values[0])
