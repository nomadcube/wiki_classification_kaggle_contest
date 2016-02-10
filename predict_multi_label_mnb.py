# coding=utf-8
import math

import numpy as np
import numpy.ma
from array import array
from scipy.sparse import csr_matrix, csc_matrix
import increment_predict
from memory_profiler import profile

from fit_multi_label_mnb import fit


def predict(x, model):
    return increment_predict.predict_label(x.tolil(), model[1].tolil(), model[0])


def top_k_argmax(one_log_likelihood, max_class_cnt):
    if not isinstance(one_log_likelihood, csr_matrix):
        raise TypeError()
    for current_cnt in xrange(max_class_cnt):
        max_value = one_log_likelihood.data.argmax()
        new_class = one_log_likelihood.indices[max_value]
        one_log_likelihood.data = numpy.ma.masked_values(one_log_likelihood.data,
                                                         value=one_log_likelihood.data[max_value])
        yield new_class


def log_likelihood(x, model):
    b, w = model
    tmp = w.dot(x.transpose()).tocsr()
    tmp.data += b.repeat(np.diff(tmp.indptr))
    return tmp


# @profile
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


if __name__ == '__main__':
    test_y = np.array([[0], [0], [1], [1], [0],
                       [0], [0], [1], [1], [1],
                       [1], [1], [1], [1], [0]])
    element = [1.] * 30
    row_index = list()
    for i in range(15):
        row_index.extend([i] * 2)
    col_index = [0, 3,
                 0, 4,
                 0, 4,
                 0, 3,
                 0, 3,
                 1, 3,
                 1, 4,
                 1, 4,
                 1, 5,
                 1, 5,
                 2, 5,
                 2, 4,
                 2, 4,
                 2, 5,
                 2, 5]
    test_x = csr_matrix((element, (row_index, col_index)), shape=(15, 6))
    m = fit(test_y, test_x)
    new_x = csr_matrix(([1., 1.], ([0, 0], [1, 3])), shape=(1, 6))
    print(predict(new_x, m, 1))
