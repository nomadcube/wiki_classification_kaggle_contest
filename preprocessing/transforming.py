import numpy as np
from array import array
from scipy.sparse import coo_matrix, csr_matrix
from tf_idf import tf_idf
from memory_profiler import profile


class XConverter:
    def __init__(self, feature_selection_percentile):
        self._percentile = feature_selection_percentile
        self._old_new_features_rel = None
        self.selected_features = None

    @profile
    def construct(self, X):
        if not isinstance(X, csr_matrix):
            raise TypeError()
        tf_idf_X = tf_idf(X)
        self.selected_features = self._pick_features(tf_idf_X.indices, tf_idf_X.data, self._percentile)
        self._old_new_features_rel = {old: new for new, old in enumerate(self.selected_features)}

    def convert(self, X):
        coo_x = X.tocoo()
        new_data = array('f')
        new_row = array('I')
        new_col = array('I')
        for i in xrange(len(coo_x.data)):
            if coo_x.col[i] in self.selected_features:
                new_data.append(coo_x.data[i])
                new_col.append(self._old_new_features_rel[coo_x.col[i]])
                new_row.append(coo_x.row[i])
        return coo_matrix((new_data, (new_row, new_col)), shape=(X.shape[0], len(self.selected_features)),
                          dtype='float').tocsr()

    @staticmethod
    def _pick_features(origin_features, scores, percentage):
        if len(origin_features) != len(scores):
            raise ValueError('features and scores must be of the same length.')
        threshold = np.percentile(scores, percentage)
        good_features = set()
        for i in xrange(len(scores)):
            if scores[i] >= threshold:
                good_features.add(origin_features[i])
        return good_features


class YConverter:
    def __init__(self):
        self.label_old_new_relation = dict()

    def construct(self, y):
        for labels in y:
            for each_label in labels:
                if each_label not in self.label_old_new_relation.keys():
                    self.label_old_new_relation[each_label] = len(self.label_old_new_relation)

    def convert(self, y):
        new_y = list()
        for labels in y:
            new_labels = array('I')
            for each_label in labels:
                new_labels.append(self.label_old_new_relation[each_label])
            new_y.append(new_labels)
        return new_y

    def withdraw_convert(self, new_y):
        new_old_relation = {new: old for old, new in self.label_old_new_relation.items()}
        y = list()
        for labels in new_y:
            new_labels = array('I')
            for each_label in labels:
                new_labels.append(new_old_relation[each_label])
            y.append(new_labels)
        return y


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
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]

    origin_x = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    x_converter = XConverter(100)
    x_converter.construct(origin_x)
    new_x = x_converter.convert(origin_x)
    print new_x

    y = [[314523, 165538, 416827], [21631], [76255, 165538]]
    y_converter = YConverter()
    y_converter.construct(y)
    print y_converter.convert(y)
    print y_converter.withdraw_convert(y_converter.convert(y))
