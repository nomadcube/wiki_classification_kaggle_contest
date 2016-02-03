from scipy.sparse import csc_matrix, coo_matrix
from array import array


def detect_redundant_feature(condition_probability_table):
    res = set()
    for col_index in xrange(condition_probability_table.shape[1]):
        column_data = condition_probability_table.data[
                      condition_probability_table.indptr[col_index]: condition_probability_table.indptr[col_index + 1]]
        if column_data.shape[0] > 1 and column_data.mean() == column_data[0]:
            res.add(col_index)
    return res


def remove_redundant_feature_and_zero(condition_probability_table):
    coo_cpt = condition_probability_table.tocoo()
    new_data = array('f')
    new_column = array('I')
    new_row = array('I')
    redundant_features = detect_redundant_feature(condition_probability_table)
    for i, non_zero in enumerate(coo_cpt.data):
        current_column = coo_cpt.col[i]
        if non_zero != 0. and current_column not in redundant_features:
            new_data.append(non_zero)
            new_row.append(coo_cpt.row[i])
            new_column.append(current_column)
    return coo_matrix((new_data, (new_row, new_column)), shape=condition_probability_table.shape, dtype='float').tocsc()


if __name__ == '__main__':
    cpt = csc_matrix(([1., 1., 1., 0., 3., 1., 1., 1.], [0, 1, 2, 0, 1, 0, 1, 2], [0, 3, 5, 8]), shape=(3, 3),
                     dtype='float')
    print cpt
    print [c for c in detect_redundant_feature(cpt)]
    print remove_redundant_feature_and_zero(cpt)
