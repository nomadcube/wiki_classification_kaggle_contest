from scipy.sparse import csr_matrix
import numpy as np
import math


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
    log_inverse_doc_frequency = np.zeros(max(feature_occurrence.keys()) + 1)
    for feature, occurrence in feature_occurrence.items():
        log_inverse_doc_frequency[feature] = math.log(float(total_doc_count) / occurrence)
    return log_inverse_doc_frequency


def tf_idf(count_mat):
    return csr_matrix(tf(count_mat).multiply(idf(count_mat)))


if __name__ == '__main__':
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    print(tf(mat))
    print(idf(mat))
    print(tf_idf(mat))
