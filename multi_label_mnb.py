import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def construct_coo_from_list(lst):
    elements = list()
    rows = list()
    columns = list()
    for row_index, row in enumerate(lst):
        row_size = len(row)
        elements.extend([1.0] * row_size)
        rows.extend([row_index] * row_size)
        columns.extend(row)
    return coo_matrix((elements, (rows, columns)), shape=(len(lst), max(columns) + 1), dtype='float')


def fit(y, x):
    coo_y = construct_coo_from_list(y)
    csr_y = coo_y.tocsr()
    label_count = csr_matrix(coo_y.sum(axis=0))
    label_feature_count = csr_y.transpose().dot(x)
    return label_count, label_feature_count


if __name__ == '__main__':
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    print(fit(test_y, test_x)[0])
    print(fit(test_y, test_x)[1])
