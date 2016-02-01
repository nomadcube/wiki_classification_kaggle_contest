# coding=utf-8
import math

import numpy as np
import numpy.ma
from array import array
from scipy.sparse import csr_matrix, csc_matrix
from memory_profiler import profile

from fit_multi_label_mnb import fit


# @profile
def predict(x, b, w, k=1):
    res = list()
    for i, each_x in enumerate(x):
        # print(i)
        res.append(_one_predict(each_x, b, w, k))
    return res


def _one_predict(one_x, b, w, k=1):
    ll = _log_likelihood(one_x, b, w)
    return [am for am in _top_k_argmax(ll, k)]


def _log_likelihood(x, b, w):
    return np.array(x.dot(w.transpose()).toarray() + b)[0]


def _top_k_argmax(arr, k):
    if not isinstance(arr, np.ndarray):
        raise TypeError()
    for i in xrange(k):
        tmp_am = arr.argmax()
        yield tmp_am
        arr = numpy.ma.masked_values(arr, value=arr[tmp_am])


def convert_to_linear(model):
    b = np.array([math.log(i) if i > 0 else 1e-8 for i in np.array(model[0])[0]])
    w = model[1].tocsc()
    w.data = np.array([math.log(i) if i > 0 else 1e-8 for i in w.data])
    return [b, w]


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
    b, w = convert_to_linear(m)
    new_x = csr_matrix(([1., 1.], ([0, 0], [1, 3])), shape=(1, 6))
    print predict(new_x, b, w, 1)
