# coding=utf-8
from array import array
from memory_profiler import profile
from scipy.sparse import csr_matrix
import numpy as np


class Sample:
    def __init__(self):
        self.y = list()
        self.x = None
        self.class_cnt = -1
        self._element = array('f')
        self._row_indptr = array('I')
        self._row_indptr.append(0)
        self._col_index = array('I')

    def _read_one_line(self, line):
        multi_label, instance = line.strip().split(' ', 1)

        all_labels = array('I', [int(l) for l in multi_label.split(',')])
        self.y.append(all_labels)
        for label in all_labels:
            self.class_cnt = label + 1 if label > self.class_cnt else self.class_cnt

        all_features = instance.split(' ')
        self._row_indptr.append(len(all_features) + self._row_indptr[-1])
        for feature in all_features:
            column, element = feature.split(':')
            column = int(column)
            self._element.append(float(element))
            self._col_index.append(column)
        return self

    def _convert_x_to_csr(self):
        self.x = csr_matrix((self._element, self._col_index, self._row_indptr),
                            shape=(len(self._row_indptr) - 1, max(self._col_index) + 1), dtype='float')

    def read(self, data_file_path):
        with open(data_file_path) as f:
            for line_no, line in enumerate(f.__iter__()):
                self._read_one_line(line)
        self._convert_x_to_csr()
        return self

    def extract_and_update(self, extract_cnt):
        if not extract_cnt < len(self.y):
            raise ValueError()
        subset = Sample()
        row_cnt, feature_dimension = self.x.shape
        subset.y = self.y[:extract_cnt]
        subset._row_indptr = self._row_indptr[:(extract_cnt + 1)]
        subset._element = self._element[:subset._row_indptr[-1]]
        subset._col_index = self._col_index[:subset._row_indptr[-1]]
        subset.x = csr_matrix((subset._element, subset._col_index, subset._row_indptr),
                              shape=(extract_cnt, feature_dimension), dtype='float')
        subset.class_cnt = self.class_cnt

        self.y = self.y[extract_cnt:]
        self._row_indptr = np.array(self._row_indptr[extract_cnt:])
        self._row_indptr -= len(subset._element)
        self._element = self._element[subset._row_indptr[-1]:]
        self._col_index = self._col_index[subset._row_indptr[-1]:]
        self.x = csr_matrix((self._element, self._col_index, self._row_indptr),
                            shape=(row_cnt - extract_cnt, feature_dimension), dtype='float')

        return subset


if __name__ == '__main__':
    smp = Sample()
    smp.read('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')
    print smp.y
    print smp.x
    print smp.class_cnt
    sub_smp = smp.extract_and_update(2)
    print sub_smp.y
    print sub_smp.x
    print sub_smp.class_cnt
