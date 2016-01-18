import itertools

from collections import namedtuple

from scipy.sparse import csr_matrix

from pympler.asizeof import asizeof


def describe(data_file_path):
    dimension = 0
    with open(data_file_path) as f:
        for line_no, line in enumerate(f):
            y, x = line.split(' ', 1)
            for feature in x.split(' '):
                col = int(feature.split(':')[0])
                if col > dimension:
                    dimension = col
    return dimension


def sample_reader(data_file_path, sample_size):
    """
    Read data_processing from data_file_path and convert it to a Sample object.

    :param data_file_path: str
    :param sample_size: int
    :return: sample
    """
    sample = namedtuple('sample', 'y x')
    y = list()
    element_x = list()
    row_index_x = list()
    col_index_x = list()
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), sample_size)):
            multi_label, instance = line.strip().split(' ', 1)
            y.append([int(label) for label in multi_label.split(',')])
            for feature in instance.split(' '):
                column, element = feature.split(':')
                element_x.append(int(element))
                col_index_x.append(int(column))
                row_index_x.append(int(line_no))
    x = csr_matrix((element_x, (row_index_x, col_index_x)), shape=(max(row_index_x) + 1, max(col_index_x) + 1),
                   dtype='int32')
    return sample(y, x)


if __name__ == '__main__':
    smp = sample_reader('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt', 2)
    print(smp.y)
    print(smp.x)
    print(smp.x.dtype)
    print(asizeof(smp.x))
