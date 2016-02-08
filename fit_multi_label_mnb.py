# coding=utf-8
import numpy as np
from array import array
from scipy.sparse import csr_matrix, csc_matrix
from memory_profiler import profile


# @profile
def fit(y, x):
    if not isinstance(x, csr_matrix):
        raise TypeError('x must be of type csr matrix.')
    y = construct_csr_from_list(y)
    y_col_sum = np.array(y.sum(axis=0))[0]
    total_label_occurrence_cnt = y_col_sum.sum()
    y_col_sum /= total_label_occurrence_cnt
    y_col_sum = np.log(y_col_sum)

    y_x_param = y.transpose().dot(x).todense()
    y_x_param += 1.
    # todo: 正在修复feature hashing的问题
    tmp = np.array(y_x_param.sum(axis=1).ravel())[0]
    y_x_param /= tmp.repeat(y_x_param.shape[1]).reshape(y_x_param.shape)
    y_x_param = np.log(y_x_param)

    return y_col_sum, csc_matrix(y_x_param)


def construct_csr_from_list(two_dimension_arr, element_dtype='float', max_n_dim=None):
    elements = array('f')
    rows = array('I')
    columns = array('I')
    for row_index, row in enumerate(two_dimension_arr):
        row_size = len(row)
        elements.extend(array('f', [1.0] * row_size))
        rows.extend(array('I', [row_index] * row_size))
        columns.extend(array('I', row))
    n_dim = max_n_dim if max_n_dim else (max(columns) + 1)
    return csr_matrix((elements, (rows, columns)), shape=(len(two_dimension_arr), n_dim), dtype=element_dtype)


if __name__ == '__main__':
    import random

    X = np.random.randint(1, 5, size=(90, 100))
    y = np.array([random.randint(1, 10) for i in range(90)])
    print y

    clf = MultinomialNB(alpha=0.)
    clf.fit(X, y)
    print clf.predict(X)

    new_y = np.array([[i] for i in y])
    m = fit(new_y, csr_matrix(X))
    import predict_multi_label

    print predict_multi_label_mnb.predict(csr_matrix(X), m, 1)
