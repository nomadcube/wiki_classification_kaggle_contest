import numpy as np
from array import array
from scipy.sparse import csr_matrix
from memory_profiler import profile


# @profile
def fit(y, x):
    if not isinstance(x, csr_matrix):
        raise TypeError('x must be of type csr matrix.')
    y = construct_csr_from_list(y)
    y_col_sum = y.sum(axis=0)
    total_label_occurrence_cnt = y.sum()
    y_param = y_col_sum / total_label_occurrence_cnt
    y_x_param = y.transpose().dot(x).tocsr()
    y_x_param.data /= np.array(y_col_sum).repeat(np.diff(y_x_param.indptr))
    return y_param, y_x_param


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
    print(fit(test_y, test_x)[0])
    print(fit(test_y, test_x)[1])
