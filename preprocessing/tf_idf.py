import numpy as np
import math
from scipy.sparse import csr_matrix


def tf_idf(count_mat):
    return _tf(count_mat).dot(_idf(count_mat))


def _counting_occurrence(arr):
    arr.sort()
    diff = np.ones(arr.shape, arr.dtype)
    diff[1:] = np.diff(arr)
    idx = np.where(diff > 0)
    vals = np.ones(idx[0].shape[0])
    vals[0:idx[0].shape[0] - 1] = np.diff(idx)[0]
    vals[-1] = arr.shape[0] - idx[0].shape[0]
    return arr[idx], vals


def _tf(count_mat):
    updated_mat = csr_matrix(count_mat.shape)
    row_sum = csr_matrix.sum(count_mat, axis=1).ravel()
    updated_mat.data = count_mat.data / np.array(row_sum.repeat(np.diff(count_mat.indptr)))[0]
    updated_mat.indptr = count_mat.indptr
    updated_mat.indices = count_mat.indices
    return updated_mat


def _idf(count_mat):
    total_doc_count = count_mat.shape[0]
    features = np.array(count_mat.indices)
    feature, occurrence = _counting_occurrence(features)
    init_row = feature
    init_element = [math.log(float(total_doc_count) / occ) for occ in occurrence]
    return csr_matrix((init_element, (init_row, init_row)), shape=(count_mat.shape[1], count_mat.shape[1]))


if __name__ == '__main__':
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_mat = tf_idf(mat)
    print tf_mat
