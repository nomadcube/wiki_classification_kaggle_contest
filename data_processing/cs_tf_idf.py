from scipy.sparse import csr_matrix
import numpy as np
import math


def cs_mat_term_freq(cs_mat):
    row_sum = csr_matrix.sum(cs_mat, axis=1).ravel()
    term_frequency = np.array(element) / np.array(row_sum.repeat(np.diff(cs_mat.indptr)))
    return csr_matrix((term_frequency.ravel(), (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))


def log_inverse_doc_freq(cs_mat):
    column_occurrence = cs_mat.indices.tolist()
    doc_num = cs_mat.shape[0]
    log_inverse_doc_frequency = np.zeros(max(column_occurrence) + 1)
    for column in column_occurrence:
        log_inverse_doc_frequency[column] = math.log(float(doc_num) / column_occurrence.count(column))
    return log_inverse_doc_frequency


def cs_mat_tf_idf(cs_mat):
    cs_mat_tf = cs_mat_term_freq(cs_mat)
    vec_idf = log_inverse_doc_freq(cs_mat)
    return cs_mat_tf.multiply(vec_idf)


if __name__ == '__main__':
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_idf_mat = csr_matrix(cs_mat_tf_idf(mat))
    print(tf_idf_mat)
