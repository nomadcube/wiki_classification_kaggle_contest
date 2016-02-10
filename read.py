# coding=utf-8
from array import array
from memory_profiler import profile
from scipy.sparse import csr_matrix


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


if __name__ == '__main__':
    smp = Sample()
    smp.read('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')
    print smp.y
    print smp.x
    print smp.class_cnt
