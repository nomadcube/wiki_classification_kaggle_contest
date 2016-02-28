# coding=utf-8
from array import array
from memory_profiler import profile
from scipy.sparse import csr_matrix
from data_analysis.labels import occurrence
from itertools import imap, ifilter, compress


class Sample:
    def __init__(self):
        self.y = list()
        self.x = None

    def __len__(self):
        return len(self.y)

    def read(self, data_file_path):
        _element = array('f')
        _row_indptr = array('I')
        _col_index = array('I')
        _row_indptr.append(0)
        with open(data_file_path) as f:
            for line_no, line in enumerate(f):
                self._read_one_line(line, _element, _row_indptr, _col_index)
        self.x = csr_matrix((_element, _col_index, _row_indptr), shape=(len(_row_indptr) - 1, max(_col_index) + 1),
                            dtype='float')
        return self

    def extract_and_update(self):
        test_instances, common_labels_cnt = self._select_instances()
        test_instances = list(test_instances)
        train_instances = list(set(range(len(self.y))).difference(test_instances))

        test_smp = Sample()
        train_smp = Sample()

        test_smp.y = [self.y[i] for i in test_instances]
        test_smp.x = self.x[test_instances, :]

        train_smp.y = [self.y[i] for i in train_instances]
        train_smp.x = self.x[train_instances, :]

        return train_smp, test_smp, common_labels_cnt

    def _read_one_line(self, line, _element, _row_indptr, _col_index):
        multi_label, instance = line.strip().split(' ', 1)

        all_labels = array('I', imap(int, multi_label.split(',')))
        self.y.append(all_labels)

        all_features = instance.replace(r':', ' ').split(' ')
        one_line_feature_len = len(all_features) / 2
        _row_indptr.append(one_line_feature_len + _row_indptr[-1])
        _element.extend([float(i) for i in compress(all_features, [0, 1] * one_line_feature_len)])
        _col_index.extend([int(i) for i in compress(all_features, [1, 0] * one_line_feature_len)])

    def _select_instances(self):
        # todo: common_labels_cnt只是个大概的估计，并不是准确的"训练集和测试集共有的label个数"
        test_instances = set()
        label_occurrence = occurrence(self.y)
        common_labels_cnt = 0.
        for label, instance_of_label in label_occurrence.items():
            if len(instance_of_label) > 2:
                for i in range(int(len(instance_of_label) * 0.5)):
                    test_instances.add(instance_of_label.pop())
                common_labels_cnt += 1.
        return test_instances, common_labels_cnt


if __name__ == '__main__':
    smp = Sample()
    smp.read('/Users/wumengling/PycharmProjects/kaggle/input_data/train_subset.csv')
    print smp.y
    print smp.x
    tr_smp, te_smp = smp.extract_and_update()
    print "================"
    print tr_smp.y
    print tr_smp.x
    print "==============="
    print te_smp.y
    print te_smp.x
