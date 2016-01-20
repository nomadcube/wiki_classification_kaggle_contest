from sparse_tf_idf import tf_idf


def test_cs_mat_tf_idf():
    import numpy as np
    from scipy.sparse import csr_matrix
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_idf_mat = csr_matrix(tf_idf(mat))
    assert tf_idf_mat.nnz == 6
    assert round(tf_idf_mat.data[0], 8) == 0.40546511
    assert round(tf_idf_mat.data[1], 8) == 0.18310205
    assert round(tf_idf_mat.data[2], 8) == 0.06757752
    assert round(tf_idf_mat.data[3], 8) == 0.73240819
    assert round(tf_idf_mat.data[4], 8) == 0.20273255
    assert round(tf_idf_mat.data[5], 8) == 0.20273255
