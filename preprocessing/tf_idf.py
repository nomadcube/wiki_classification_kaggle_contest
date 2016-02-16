import numpy as np
import math
from scipy.sparse import csr_matrix


def tf_idf(count_mat):
    return _tf(count_mat).dot(_idf(count_mat))


def _counting_occurrence(array_like):
    occurrence = dict()
    for each_element in array_like:
        occurrence.setdefault(each_element, 0)
        occurrence[each_element] += 1
    return occurrence


def _tf(count_mat):
    updated_mat = csr_matrix(count_mat.shape)
    row_sum = csr_matrix.sum(count_mat, axis=1).ravel()
    updated_mat.data = count_mat.data / np.array(row_sum.repeat(np.diff(count_mat.indptr)))[0]
    updated_mat.indptr = count_mat.indptr
    updated_mat.indices = count_mat.indices
    return updated_mat


def _idf(count_mat):
    total_doc_count = count_mat.shape[0]
    feature_occurrence = _counting_occurrence(count_mat.indices)
    init_row = list()
    init_element = list()
    for feature, occurrence in feature_occurrence.items():
        init_row.append(feature)
        init_element.append(math.log(float(total_doc_count) / occurrence))
    return csr_matrix((init_element, (init_row, init_row)), shape=(count_mat.shape[1], count_mat.shape[1]))


if __name__ == '__main__':
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_mat = tf_idf(mat)
    print tf_mat
