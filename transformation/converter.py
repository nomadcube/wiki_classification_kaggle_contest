from array import array

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from feature_selection.tf_idf import tf_idf


class XConverter:
    def __init__(self, x, feature_selection_percentile):
        tf_idf_x = tf_idf(x)
        self._percentile = feature_selection_percentile
        self.selected_features = self._pick_features(tf_idf_x.indices, tf_idf_x.data, self._percentile)
        self._old_new_features_rel = {old: new for new, old in enumerate(self.selected_features)}
        self._percentile = feature_selection_percentile

    def convert(self, x):
        coo_x = x.tocoo()
        new_data = array('f')
        new_row = array('I')
        new_col = array('I')
        for i in xrange(len(coo_x.data)):
            current_feature = coo_x.col[i]
            if current_feature in self.selected_features:
                new_data.append(coo_x.data[i])
                new_col.append(self._old_new_features_rel[current_feature])
                new_row.append(coo_x.row[i])
        return coo_matrix((new_data, (new_row, new_col)), shape=(x.shape[0], len(self.selected_features)),
                          dtype='float').tocsr()

    @staticmethod
    def _pick_features(origin_features, scores, percentage):
        threshold = np.percentile(scores, percentage)
        good_indices = np.where(scores >= threshold)
        return set(origin_features[good_indices])


def convert_y_to_csr(y):
    elements = array('f')
    instance_nos = array('I')
    labels = array('I')
    for row_index, row in enumerate(y):
        row_size = len(row)
        elements.extend(array('f', [1.0] * row_size))
        instance_nos.extend(array('I', [row_index] * row_size))
        labels.extend(array('I', row))
    return csr_matrix((elements, (labels, instance_nos)), dtype='float')
