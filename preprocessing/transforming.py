import numpy as np
from array import array
from scipy.sparse import coo_matrix, csr_matrix
from tf_idf import tf_idf
import math
from memory_profiler import profile
from itertools import chain


class XConverter:
    def __init__(self, feature_selection_percentile):
        self._percentile = feature_selection_percentile
        self._old_new_features_rel = None
        self.selected_features = None

    # @profile
    def construct(self, x):
        tf_idf_x = tf_idf(x)
        self.selected_features = self._pick_features(tf_idf_x.indices, tf_idf_x.data, self._percentile)
        self._old_new_features_rel = {old: new for new, old in enumerate(self.selected_features)}

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


class YConverter:
    def __init__(self):
        self.label_old_new_relation = dict()

    def construct(self, y):
        a = np.unique([each_y for each_y in chain.from_iterable(y)])
        self.label_old_new_relation = dict(zip(a, range(len(a))))

    def convert(self, y):
        new_y = list()
        for i, labels in enumerate(y):
            new_y.append(array('I', [self.label_old_new_relation[each_label] for j, each_label in enumerate(labels)]))
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


def convert_y_to_csr(y, element_dtype='float', total_label_cnt=0):
    elements = array('f')
    instance_nos = array('I')
    labels = array('I')
    for row_index, row in enumerate(y):
        row_size = len(row)
        elements.extend(array('f', [1.0] * row_size))
        instance_nos.extend(array('I', [row_index] * row_size))
        labels.extend(array('I', row))
    cnt_label = max(total_label_cnt, max(labels) + 1)
    return csr_matrix((elements, (labels, instance_nos)), shape=(cnt_label, len(y)), dtype=element_dtype)


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
    mapped_csr_y = convert_y_to_csr(y_converter.convert(y))
    print mapped_csr_y
    all_part = [i for i in part_csr_y_generator(mapped_csr_y, 1)]
    for i in all_part:
        print i[0]
        print i[1]
