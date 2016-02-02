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

    y_x_param = y.transpose().dot(x).tocsr()
    y_x_param.data /= y_col_sum.repeat(np.diff(y_x_param.indptr))

    total_label_occurrence_cnt = y_col_sum.sum()
    y_col_sum /= total_label_occurrence_cnt

    return combine_b_w(y_col_sum, y_x_param.tocsc())


def construct_csr_from_list(two_dimension_arr, max_n_dim=None):
    elements = array('f')
    rows = array('I')
    columns = array('I')
    for row_index, row in enumerate(two_dimension_arr):
        row_size = len(row)
        elements.extend(array('f', [1.0] * row_size))
        rows.extend(array('I', [row_index] * row_size))
        columns.extend(array('I', row))
    n_dim = max_n_dim if max_n_dim else (max(columns) + 1)
    return csr_matrix((elements, (rows, columns)), shape=(len(two_dimension_arr), n_dim), dtype='float')


# @profile
def combine_b_w(b, w):
    new_data = array('f', w.data)
    new_indices = array('I', w.indices)
    new_indptr = array('I', w.indptr)
    current_feature_cnt = w.nnz
    nnz_b = 0
    for b_index, each_b in enumerate(b):
        if each_b > 0.:
            nnz_b += 1
            new_indices.append(b_index)
            new_data.append(each_b)
    new_indptr.append(current_feature_cnt + nnz_b)
    combined_m = csc_matrix((new_data, new_indices, new_indptr), shape=(w.shape[0], w.shape[1] + 1), dtype='float')
    combined_m.data = np.log(combined_m.data)
    return combined_m


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
    print(test_x)
    print('========')
    print(fit(test_y, test_x))
