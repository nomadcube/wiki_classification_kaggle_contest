import math

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from cpp_ext import k_argmax


def fit(y, x):
    if not isinstance(x, csr_matrix):
        raise TypeError('x must be of type csr matrix.')
    y = construct_csr_from_list(y)
    total_label_occurrence_cnt = y.sum()
    new_x = log_rate_per_row(x)
    new_x = add_unit_column(new_x, total_label_occurrence_cnt)
    return y.transpose().dot(new_x)


def construct_csr_from_list(two_dimension_list, max_n_dim=None):
    elements = list()
    rows = list()
    columns = list()
    for row_index, row in enumerate(two_dimension_list):
        row_size = len(row)
        elements.extend([1.0] * row_size)
        rows.extend([row_index] * row_size)
        columns.extend(row)
    n_dim = max_n_dim if max_n_dim else (max(columns) + 1)
    return csr_matrix((elements, (rows, columns)), shape=(len(two_dimension_list), n_dim), dtype='float')


def log_rate_per_row(mat):
    if not isinstance(mat, csr_matrix):
        raise TypeError('freq_mat must be of csr_matrix type.')
    if mat.dtype != 'float':
        raise TypeError('dtype of freq_mat must be of float.')
    new_mat = mat.copy()
    row_sum = new_mat.sum(axis=1)
    new_mat.data /= np.array(row_sum.repeat(np.diff(new_mat.indptr)))[0]
    new_mat.data = np.log(new_mat.data)
    return new_mat


def add_unit_column(mat, coefficient=None):
    mat = mat.tocsc()
    n_row = mat.shape[0]
    current_nnz = mat.nnz
    dat = mat.data.tolist()
    if coefficient:
        dat.extend([math.log(1. / coefficient)] * n_row)
    else:
        dat.extend([1.] * n_row)
    indi = mat.indices.tolist()
    indi.extend(range(n_row))
    indp = mat.indptr.tolist()
    indp.append(current_nnz + n_row)
    return csc_matrix((dat, indi, indp), shape=(mat.shape[0], mat.shape[1] + 1), dtype='float').tocsr()


if __name__ == '__main__':
    print(k_argmax.k_argmax([0.1, 1.1, 2.1, 3.2], [3, 1, 0, 2], [0, 1, 4, 4]))
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    print(test_x)
    print('========')
    print(add_unit_column(test_x, 1.))
    print('========')
    print(fit(test_y, test_x, 6))
