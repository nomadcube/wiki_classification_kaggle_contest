from scipy.sparse import csr_matrix
import numpy as np
import math

element = np.array([1., 1., 1., 4., 1., 1.])
row_index = np.array([0, 1, 1, 1, 2, 2])
col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]

mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
row_sum = csr_matrix.sum(mat, axis=1).ravel()
term_freq = np.array(element) / np.array(row_sum.repeat(np.diff(mat.indptr)))
term_freq_mat = csr_matrix((term_freq.ravel(), (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))

doc_num = term_freq_mat.shape[0]
log_inverse_doc_freq = list()
for col in col_index:
    log_inverse_doc_freq.append(math.log(float(doc_num) / col_index.count(col)))
log_inverse_doc_freq_mat = csr_matrix((log_inverse_doc_freq, ([0] * len(log_inverse_doc_freq), col_index)),
                                      shape=(1, max(col_index) + 1))
# print(log_inverse_doc_freq_mat)
# print(term_freq_mat)
for row in range(max(row_index) + 1):
    print(term_freq_mat.getrow(row).multiply(log_inverse_doc_freq_mat))
print(term_freq_mat)
