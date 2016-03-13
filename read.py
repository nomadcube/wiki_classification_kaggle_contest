# coding=utf-8
import math
from array import array
from memory_profiler import profile
from scipy.sparse import csr_matrix
from itertools import compress

from data_analysis.labels import occurrence


class Sample:
    def __init__(self):
        self.y = list()
        self.x = None

    def __len__(self):
        return len(self.y)

    def read(self, data_file_path, target_class='24177'):
        _element = array('f')
        _row_indptr = array('I')
        _col_index = array('I')
        _row_indptr.append(0)
        with open(data_file_path) as f:
            for line_no, line in enumerate(f):
                self._read_one_line(line, _element, _row_indptr, _col_index, target_class)
        self.x = csr_matrix((_element, _col_index, _row_indptr), shape=(len(_row_indptr) - 1, max(_col_index) + 1),
                            dtype='float')
        return self

    def _read_one_line(self, line, _element, _row_indptr, _col_index, target_class):
        multi_label, instance = line.strip().split(' ', 1)

        self.y.append(1) if target_class in multi_label.split(',') else self.y.append(0)

        all_features = instance.replace(r':', ' ').split(' ')
        one_line_feature_len = len(all_features) / 2
        _row_indptr.append(one_line_feature_len + _row_indptr[-1])
        _element.extend([float(i) for i in compress(all_features, [0, 1] * one_line_feature_len)])
        _col_index.extend([int(i) for i in compress(all_features, [1, 0] * one_line_feature_len)])

    def extract_and_update(self):
        test_instances = self._select_instances()
        test_instances = list(test_instances)
        train_instances = list(set(range(len(self.y))).difference(test_instances))

        test_smp = Sample()
        train_smp = Sample()

        test_smp.y = [self.y[i] for i in test_instances]
        test_smp.x = self.x[test_instances, :]

        train_smp.y = [self.y[i] for i in train_instances]
        train_smp.x = self.x[train_instances, :]

        return train_smp, test_smp

    def _select_instances(self):
        test_instances = set()
        label_occurrence = occurrence(self.y)
        for label, instance_of_label in label_occurrence.items():
            if len(instance_of_label) > 1:
                num_in_cv = int(math.ceil(len(instance_of_label) * 0.2))
                if num_in_cv > 0:
                    for i in range(num_in_cv):
                        test_instances.add(instance_of_label.pop())
        return test_instances
