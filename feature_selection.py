from scipy.sparse import csc_matrix, coo_matrix
from array import array
from memory_profiler import profile


# @profile
def redundant_feature(condition_probability_table, col_index):
    column_data = condition_probability_table.data[
                  condition_probability_table.indptr[col_index]: condition_probability_table.indptr[col_index + 1]]
    return column_data.shape[0] > 1 and column_data.mean() == column_data[0]


# @profile
def remove_redundant_feature_and_zero(condition_probability_table):
    coo_cpt = condition_probability_table.tocoo()
    new_data = array('f')
    new_column = array('I')
    new_row = array('I')
    for i, non_zero in enumerate(coo_cpt.data):
        current_column = coo_cpt.col[i]
        if not redundant_feature(condition_probability_table, current_column):
            new_data.append(non_zero)
            new_row.append(coo_cpt.row[i])
            new_column.append(current_column)
    return coo_matrix((new_data, (new_row, new_column)), shape=condition_probability_table.shape, dtype='float').tocsc()


if __name__ == '__main__':
    cpt = csc_matrix(([1., 1., 1., 2., 3., 1., 1., 2.], [0, 1, 2, 0, 1, 0, 1, 2], [0, 3, 5, 8]), shape=(3, 3),
                     dtype='float')
    # print cpt
    print remove_redundant_feature_and_zero(cpt)
