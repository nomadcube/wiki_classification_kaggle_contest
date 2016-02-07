from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from array import array
import sparse_tf_idf
import numpy as np


def high_tf_idf_features(tf_idf_mat, threshold):
    good_features = set()
    if not isinstance(tf_idf_mat, csr_matrix):
        raise TypeError()
    for i in xrange(len(tf_idf_mat.data)):
        if tf_idf_mat.data[i] >= threshold:
            good_features.add(tf_idf_mat.indices[i])
    return good_features


def construct_lower_rank_x(original_x, good_features):
    coo_x = original_x.tocoo()
    new_data = array('f')
    new_row = array('I')
    new_col = array('I')
    for i in xrange(len(coo_x.data)):
        if coo_x.col[i] in good_features:
            new_data.append(coo_x.data[i])
            new_col.append(coo_x.col[i])
            new_row.append(coo_x.row[i])
    return coo_matrix((new_data, (new_row, new_col)), shape=original_x.shape, dtype='float')


if __name__ == '__main__':
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_mat = sparse_tf_idf.tf_idf(mat)
    print tf_mat
    gf = high_tf_idf_features(tf_mat, 0.5)
    print construct_lower_rank_x(mat, gf).shape
