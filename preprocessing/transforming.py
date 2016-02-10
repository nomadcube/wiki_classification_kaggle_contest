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


def feature_mapping(origin_x, old_new_relation):
    if not isinstance(origin_x, csr_matrix):
        raise TypeError()
    coo_x = origin_x.tocoo()
    coo_x.col = array('I', [old_new_relation[c] for c in coo_x.col])
    return origin_x


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
