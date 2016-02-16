import numpy as np
from array import array
from scipy.sparse import coo_matrix, csr_matrix


def pick_features(features, scores, percentage):
    if len(features) != len(scores):
        raise ValueError('features and scores must be of the same length.')
    threshold = np.percentile(scores, percentage)
    good_features = set()
    for i in xrange(len(scores)):
        if scores[i] >= threshold:
            good_features.add(features[i])
    return good_features


def dimension_reduction(origin_x, selected_features):
    coo_x = origin_x.tocoo()
    new_data = array('f')
    new_row = array('I')
    new_col = array('I')
    for i in xrange(len(coo_x.data)):
        if coo_x.col[i] in selected_features:
            new_data.append(coo_x.data[i])
            new_col.append(coo_x.col[i])
            new_row.append(coo_x.row[i])
    return coo_matrix((new_data, (new_row, new_col)), shape=(origin_x.shape[0], max(new_col) + 1),
                      dtype='float').tocsr()


def feature_mapping(origin_x, features):
    old_new_relation = {old: new for new, old in enumerate(features)}
    if not isinstance(origin_x, csr_matrix):
        raise TypeError()
    coo_x = origin_x.tocoo()
    coo_x.col = np.array([old_new_relation[c] for c in coo_x.col], dtype='int')
    return coo_matrix((coo_x.data, (coo_x.row, coo_x.col)), shape=(origin_x.shape[0], max(coo_x.col) + 1),
                      dtype='float').tocsr()


def convert_y_to_csr(y, element_dtype='float', max_n_dim=None):
    elements = array('f')
    rows = array('I')
    columns = array('I')
    for row_index, row in enumerate(y):
        row_size = len(row)
        elements.extend(array('f', [1.0] * row_size))
        rows.extend(array('I', [row_index] * row_size))
        columns.extend(array('I', row))
    n_dim = max_n_dim if max_n_dim else (max(columns) + 1)
    return csr_matrix((elements, (rows, columns)), shape=(len(y), n_dim), dtype=element_dtype)


if __name__ == '__main__':
    from tf_idf import tf_idf

    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    origin_x = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_idf_x = tf_idf(origin_x)
    features = pick_features(tf_idf_x.indices, tf_idf_x.data, 100)
    reduced_x = dimension_reduction(origin_x, features)
    mapped_x = feature_mapping(reduced_x, features)
    print mapped_x
    print mapped_x.shape
