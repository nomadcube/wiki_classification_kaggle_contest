from pickle import dump, load


def save_with_protocol2(param, dir, file_name):
    with open('{0}/{1}'.format(dir, file_name), 'wb') as f:
        dump(param, f, protocol=2)


def load_with_protocol2(dir, file_name):
    with open('{0}/{1}'.format(dir, file_name), 'r') as f:
        return load(f)
