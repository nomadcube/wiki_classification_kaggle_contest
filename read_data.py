from array import array
from scipy.sparse import csr_matrix
from itertools import compress

from Sample import Sample


def read_sparse(data_file_path):
    smp = Sample()

    element = array('f')
    row_index_pointer = array('I')
    column_index = array('I')
    row_index_pointer.append(0)

    with open(data_file_path) as f:
        for line_no, line in enumerate(f):
            _one_read_sparse(line, smp.y, element, row_index_pointer, column_index)
    smp.x = csr_matrix((element, column_index, row_index_pointer),
                       shape=(len(row_index_pointer) - 1, max(column_index) + 1),
                       dtype='float')
    return smp


def _one_read_sparse(line, y, element, row_index_pointer, column_index):
    multi_label, instance = line.strip().split(' ', 1)
    all_features = instance.replace(r':', ' ').split(' ')
    one_line_feature_len = len(all_features) / 2

    for each_label in multi_label.split(','):
        y.append(each_label)
        row_index_pointer.append(one_line_feature_len + row_index_pointer[-1])
        element.extend([float(i) for i in compress(all_features, [0, 1] * one_line_feature_len)])
        column_index.extend([int(i) for i in compress(all_features, [1, 0] * one_line_feature_len)])
