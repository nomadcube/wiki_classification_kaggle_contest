import math

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from fit_multi_label_mnb import fit


def predict(x, model, block_size=1):
    res = list()
    for i, block_x in enumerate(_block_x(x, block_size)):
        print(i)
        res.extend(_block_predict(block_x, model))
    return res


def _block_predict(block_sample, model):
    block_res = list()
    ll = _log_likelihood(block_sample, model).tocsc()
    for block_row_index in range(ll.shape[1]):
        ll_col = ll.getcol(block_row_index)
        row_pred = [ll_col.indices[np.array(ll_col.data).argmax()]] if len(ll_col.data) > 0 else []
        block_res.append(row_pred)
    return block_res


def _block_x(x, block_size):
    n_row, n_col = x.shape
    if n_row % block_size != 0:
        raise ValueError()
    for start_row_index in range(0, (n_row - block_size + 1), block_size):
        yield x._get_submatrix(slice(start_row_index, start_row_index + block_size), slice(0, n_col))


def _log_likelihood(x, model):
    new_x = add_unit_column(x)
    return model.dot(new_x.transpose())


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
    indp = mat.indptr.tolist()
    indp.append(current_nnz + n_row)
    return csc_matrix((dat, indi, indp), shape=(mat.shape[0], mat.shape[1] + 1), dtype='float').tocsr()


def convert_to_linear_classifier(model):
    prior = list(np.array(model[0])[0])
    mn_param = model[1].tocsc()
    dat = mn_param.data.tolist()
    dat.extend(prior)
    log_dat = [math.log(d) if d > 0 else -1e10 for d in dat]
    indi = mn_param.indices.tolist()
    indi.extend(range(mn_param.shape[0]))
    indt = mn_param.indptr.tolist()
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
    print(predict(new_x, lm))
