import numpy as np
from scipy.sparse import csr_matrix

from fit_multi_label_mnb import fit


def predict(x, model, block_size=1):
    res = list()
    for block_x in _block_x(x, block_size):
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
    multinomial_parameters = model[1]
    class_prior = model[0]
    if not isinstance(x, csr_matrix):
        raise TypeError()
    if not isinstance(multinomial_parameters, csr_matrix):
        raise TypeError()
    if not isinstance(class_prior, csr_matrix):
        raise TypeError()
    log_likelihood_mat = multinomial_parameters.dot(x.transpose())
    prior_prob = class_prior.transpose().toarray()
    if len(log_likelihood_mat.data) > 0:
        log_likelihood_mat.data += np.array(prior_prob.repeat(np.diff(log_likelihood_mat.indptr)))[0]
        return log_likelihood_mat
    else:
        return csr_matrix(prior_prob.repeat(x.shape[0]))


if __name__ == '__main__':
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    m = fit(test_y, test_x)
    print(predict(test_x, m))
