# coding=utf-8
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from array import array
import numpy as np


def high_tf_idf_features(tf_idf_mat, threshold):
    threshold = np.percentile(tf_idf_mat.data, threshold)
    good_features = set()
    if not isinstance(tf_idf_mat, csr_matrix):
        raise TypeError()
    for i in xrange(len(tf_idf_mat.data)):
        if tf_idf_mat.data[i] >= threshold:
            good_features.add(tf_idf_mat.indices[i])
    return good_features


def construct_lower_rank_x(original_x, good_features, feature_mapping):
    coo_x = original_x.tocoo()
    new_data = array('f')
    new_row = array('I')
    new_col = array('I')
    for i in xrange(len(coo_x.data)):
        if coo_x.col[i] in good_features:
            new_data.append(coo_x.data[i])
            new_col.append(feature_mapping[coo_x.col[i]])
            new_row.append(coo_x.row[i])
    return coo_matrix((new_data, (new_row, new_col)), shape=(original_x.shape[0], max(new_col)), dtype='float')


def convert_y_to_csr(y, element_dtype='float', max_n_dim=None):
    elements = array('f')
    rows = array('I')
    columns = array('I')
    for row_index, row in enumerate(y):
        row_size = len(row)
        elements.extend(array('f', [1.0] * row_size))
        rows.extend(array('I', [row_index] * row_size))
        columns.extend(array('I', row))
    n_dim = max_n_dim if max_n_dim else (max(columns) + 1)
    return csr_matrix((elements, (rows, columns)), shape=(len(y), n_dim), dtype=element_dtype)


def add_unit_column(x):
    x = x.tocsc()

    new_data = array('f', x.data)
    new_indices = array('I', x.indices)
    new_indptr = array('I', x.indptr)

    current_row_cnt = x.shape[0]
    current_nnz = x.nnz

    new_data.extend(array('f', [1.] * current_row_cnt))
    new_indices.extend(array('I', xrange(current_row_cnt)))
    new_indptr.append(current_nnz + current_row_cnt)
    return csc_matrix((new_data, new_indices, new_indptr), shape=(x.shape[0], x.shape[1] + 1), dtype='float')


import math

import numpy as np
from scipy.sparse import csr_matrix


def counting_occurrence(array_like):
    occurrence = dict()
    for each_element in array_like:
        occurrence.setdefault(each_element, 0)
        occurrence[each_element] += 1
    return occurrence


def tf(count_mat):
    row_sum = csr_matrix.sum(count_mat, axis=1).ravel()
    count_mat.data /= np.array(row_sum.repeat(np.diff(count_mat.indptr)))[0]
    return count_mat


def idf(count_mat):
    total_doc_count = count_mat.shape[0]
    feature_occurrence = counting_occurrence(count_mat.indices)
    init_row = list()
    init_element = list()
    for feature, occurrence in feature_occurrence.items():
        init_row.append(feature)
        init_element.append(math.log(float(total_doc_count) / occurrence))
    return csr_matrix((init_element, (init_row, init_row)), shape=(max(init_row) + 1, max(init_row) + 1))


def tf_idf(count_mat):
    return tf(count_mat).dot(idf(count_mat))


if __name__ == '__main__':
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    print(tf_idf(mat))

if __name__ == '__main__':
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_mat = sparse_tf_idf.tf_idf(mat)
    print tf_mat
    gf = high_tf_idf_features(tf_mat, 75)
    print gf
    print construct_lower_rank_x(mat, gf)
    print mat
