# coding=utf-8
import math

import numpy as np
import numpy.ma
from array import array
from scipy.sparse import csr_matrix, csc_matrix
from memory_profiler import profile

from fit_multi_label_mnb import fit


def predict(x, model, k=1):
    res = list()
    new_x = add_unit_column(x)
    for i, block_x in enumerate(new_x):
        # print(i)
        res.append(_one_predict(block_x, model, k))
    return res


def _one_predict(one_x, model, k=1):
    ll = _log_likelihood(one_x, model)
    k_am = _top_k_argmax(np.array(ll.data), k)
    return [ll.indices[am] for am in k_am] if len(ll.data) > 0 else []


def _top_k_argmax(arr, k):
    if not isinstance(arr, np.ndarray):
        raise TypeError()
    for i in xrange(k):
        tmp_am = arr.argmax()
        yield tmp_am
        arr = numpy.ma.masked_values(arr, value=arr[tmp_am])


def _log_likelihood(x, model):
    return model.dot(x.transpose())


# @profile
def add_unit_column(mat, coefficient=None):
    mat = mat.tocsc()
    n_row = mat.shape[0]
    current_nnz = mat.nnz
    dat = mat.data.tolist()
    if coefficient:
        dat.extend([math.log(1. / coefficient)] * n_row)
    else:
        dat.extend([1.] * n_row)
    indi = mat.indices.tolist()
    indi.extend(range(n_row))
    indp = array('I', mat.indptr)
    indp.append(current_nnz + n_row)
    return csc_matrix((dat, indi, indp), shape=(mat.shape[0], mat.shape[1] + 1), dtype='float')


# @profile
def convert_to_linear_classifier(model):
    prior = array('f', np.array(model[0])[0])
    mn_param = model[1].tocsc()
    dat = array('f', mn_param.data)
    dat.extend(prior)
    log_dat = [math.log(d) if d > 0 else -1e10 for d in dat]
    indi = array('I', mn_param.indices)
    indi.extend(array('I', xrange(mn_param.shape[0])))
    indt = array('I', mn_param.indptr)
    indt.append(mn_param.nnz + mn_param.shape[0])
    return csc_matrix((log_dat, indi, indt), shape=(mn_param.shape[0], mn_param.shape[1] + 1), dtype='float')


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
    lm = convert_to_linear_classifier(m)
    new_x = csr_matrix(([1., 1.], ([0, 0], [1, 3])), shape=(1, 6))
    # print(predict(new_x, lm))
    a = np.array([1, 4, 5, 3, 9])
    print([l for l in _top_k_argmax(a, 2)])
