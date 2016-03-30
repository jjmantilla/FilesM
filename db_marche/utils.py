def sane_str(word):
    return word.replace(' ', '_').lower()


def sane_update(dic, items):
    for k, v in items.items():
        dic[sane_str(k)] = v

