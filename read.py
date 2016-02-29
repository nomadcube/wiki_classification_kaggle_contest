# coding=utf-8
import math
from array import array
from memory_profiler import profile
from scipy.sparse import csr_matrix
from itertools import imap, ifilter, compress

from data_analysis.labels import occurrence


class Sample:
    def __init__(self, mod):
        self.y = list()
        self.x = None
        self._mod = mod

    def __len__(self):
        return len(self.y)

    def read(self, data_file_path):
        """
        :param data_file_path:string
        :return: Sample对象

        分两种模式：
        1. model_selection模式，这时候需要将multi-label展开，对应的x有重，便于切分train,cv集
        2. submission模式，这时候不需要展开，否则会加大不必要的预测计算量
        """
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

        test_smp = Sample(self._mod)
        train_smp = Sample(self._mod)

        test_smp.y = [self.y[i] for i in test_instances]
        test_smp.x = self.x[test_instances, :]

        train_smp.y = [self.y[i] for i in train_instances]
        train_smp.x = self.x[train_instances, :]

        return train_smp, test_smp, common_labels_cnt

    def _read_one_line(self, line, _element, _row_indptr, _col_index):
        """
        将输入文件中的任意一行作处理后追加到输出对象中
        现在改成将multi-label展开，对应的x重复地加入到self.x中
        """
        if self._mod == 'model_selection':
            multi_label, instance = line.strip().split(' ', 1)

            all_labels = array('I', imap(int, multi_label.split(',')))
            self.y.extend([[label] for label in all_labels])
            num_repeat = len(all_labels)

            all_features = instance.replace(r':', ' ').split(' ')
            one_line_feature_len = len(all_features) / 2

            _element.extend([float(i) for i in compress(all_features, [0, 1] * one_line_feature_len)] * num_repeat)
            _col_index.extend([int(i) for i in compress(all_features, [1, 0] * one_line_feature_len)] * num_repeat)

            for repeat_time in xrange(num_repeat):
                _row_indptr.extend([one_line_feature_len + _row_indptr[-1]])

        if self._mod == 'submission':
            multi_label, instance = line.strip().split(' ', 1)

            all_labels = array('I', imap(int, multi_label.split(',')))
            self.y.append(all_labels)

            all_features = instance.replace(r':', ' ').split(' ')
            one_line_feature_len = len(all_features) / 2
            _row_indptr.append(one_line_feature_len + _row_indptr[-1])
            _element.extend([float(i) for i in compress(all_features, [0, 1] * one_line_feature_len)])
            _col_index.extend([int(i) for i in compress(all_features, [1, 0] * one_line_feature_len)])

    def _select_instances(self):
        """
        和extract_and_update一起，用于分割train, cv集
        若某个label所包含的instance数为m，
        1. 若m=1, 则该label只会在train中出现
        2. 若m > 1, 则其中的ceil(m * 0.2)个放入cv集，其它的放入train集
        """
        test_instances = set()
        label_occurrence = occurrence(self.y)
        common_labels_cnt = 0.
        for label, instance_of_label in label_occurrence.items():
            if instance_of_label > 1:
                num_in_cv = int(math.ceil(len(instance_of_label) * 0.2))
                if num_in_cv > 0:
                    for i in range(num_in_cv):
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
