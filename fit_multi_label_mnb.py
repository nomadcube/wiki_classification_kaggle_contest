import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from cpp_ext import k_argmax


def fit(y, x):
    label_count, label_feature_count = _count_occurrence(y, x)
    return _log_rate_per_column(label_count), _log_rate_per_row(label_feature_count)


def _construct_coo_from_list(lst, max_n_dim=None):
    elements = list()
    rows = list()
    columns = list()
    for row_index, row in enumerate(lst):
        row_size = len(row)
        elements.extend([1.0] * row_size)
        rows.extend([row_index] * row_size)
        columns.extend(row)
    n_dim = max_n_dim if max_n_dim else (max(columns) + 1)
    return coo_matrix((elements, (rows, columns)), shape=(len(lst), n_dim), dtype='float')


def _count_occurrence(y, x):
    coo_y = _construct_coo_from_list(y)
    csr_y = coo_y.tocsr()
    label_occurrence = csr_matrix(coo_y.sum(axis=0))
    label_feature_co_occurrence = csr_y.transpose().dot(x)
    return label_occurrence, csr_matrix(label_feature_co_occurrence)


def _log_rate_per_row(co_occurrence_frequency):
    if not isinstance(co_occurrence_frequency, csr_matrix):
        raise TypeError('freq_mat must be of csr_matrix type.')
    if co_occurrence_frequency.dtype != 'float':
        raise TypeError('dtype of freq_mat must be of float.')
    row_sum = co_occurrence_frequency.sum(axis=1)
    co_occurrence_frequency.data /= np.array(row_sum.repeat(np.diff(co_occurrence_frequency.indptr)))[0]
    co_occurrence_frequency.data = np.log(co_occurrence_frequency.data)
    return co_occurrence_frequency


def _log_rate_per_column(y_count):
    if not isinstance(y_count, csr_matrix):
        raise TypeError()
    column_sum = y_count.sum(axis=1)
    y_count.data /= np.array(column_sum)[0]
    y_count.data = np.log(y_count.data)
    return y_count


if __name__ == '__main__':
    print(k_argmax.k_argmax([0.1, 1.1, 2.1, 3.2], [3, 1, 0, 2], [0, 1, 4, 4]))
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    m = fit(test_y, test_x)
