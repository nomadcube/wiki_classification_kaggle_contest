import itertools


def get_chuck(dat_iter, chuck_size):
    chuck = [dat_slice for dat_slice in itertools.islice(dat_iter, chuck_size)]
    return chuck


def iter_chuck(y_iter, x_iter, chuck_size):
    y_chuck = get_chuck(y_iter, chuck_size)
    x_chuck = get_chuck(x_iter, chuck_size)
    while len(y_chuck):
        yield y_chuck, x_chuck
        y_chuck = get_chuck(y_iter, chuck_size)
        x_chuck = get_chuck(x_iter, chuck_size)


if __name__ == '__main__':
    x = range(8)
    y = [i * 2 for i in range(8)]
    for c_y, c_x in iter_chuck(y.__iter__(), x.__iter__(), 4):
        print('c_y: {0}; c_x: {1}'.format(c_y, c_x))
