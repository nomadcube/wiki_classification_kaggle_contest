import tf_idf


def test_tf_idf():
    import numpy as np
    from scipy.sparse import csr_matrix
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_idf_mat = tf_idf.tf_idf(mat)
    assert tf_idf_mat.nnz == 6
    assert round(tf_idf_mat[0, 1250536], 12) == 0.405465108108
    assert round(tf_idf_mat[1, 1095476], 12) == 0.732408192445
    assert round(tf_idf_mat[1, 805104], 12) == 0.067577518018
    assert round(tf_idf_mat[1, 634175], 12) == 0.183102048111
    assert round(tf_idf_mat[2, 1250536], 12) == 0.202732554054
    assert round(tf_idf_mat[2, 805104], 12) == 0.202732554054
