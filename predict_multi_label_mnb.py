import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


def predict(x, model):
    res = list()
    for sample_index in range(x.shape[0]):
        print(sample_index)
        res.append(_one_predict(x.getrow(sample_index), model))
    return res


def _one_predict(one_sample, model):
    if len(one_sample.data) > 0:
        log_likelihood = _log_likelihood(one_sample, model).tocsc()
        if not isinstance(log_likelihood, csc_matrix):
            raise TypeError()
        return [log_likelihood.indices[np.array(log_likelihood.data).argmax()]] if len(log_likelihood.data) > 0 else []
    else:
        return []


def _get_block_sample(x, start_row, end_row):
    n = x.shape[1]
    return x._get_submatrix(slice(start_row, end_row), slice(0, n))


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
