# coding=utf-8
import array
import itertools
from memory_profiler import profile
from scipy.sparse import csr_matrix


class Sample:
    def __init__(self):
        self.y = list()
        self.element_x = array.array('f')
        self.row_indptr_x = array.array('I')
        self.row_indptr_x.append(0)
        self.col_index_x = array.array('I')
        self.max_feature = -1
        self.max_class_label = -1

    def increment_by_one_line(self, line):
        multi_label, instance = line.strip().split(' ', 1)

        all_labels = array.array('I', [[int(l) for l in multi_label.split(',')][0]])
        self.y.append(all_labels)
        # todo: 只选了一个label，记得改回来
        for label in all_labels:
            self.max_class_label = label if label > self.max_class_label else self.max_class_label

        all_features = instance.split(' ')
        self.row_indptr_x.append(len(all_features) + self.row_indptr_x[-1])
        for feature in all_features:
            column, element = feature.split(':')
            column = int(column)
            self.element_x.append(float(element))
            self.col_index_x.append(column)
            self.max_feature = column if column > self.max_feature else self.max_feature
        return self

    def convert_to_csr(self, feature_dim):
        return csr_matrix((self.element_x, self.col_index_x, self.row_indptr_x),
                          shape=(len(self.row_indptr_x) - 1, feature_dim), dtype='float')


# @profile
def read_sample(data_file_path, train_size, test_size):
    test_sample = Sample()
    train_sample = Sample()
    part_train_sample = Sample()
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), 0, test_size)):
            test_sample.increment_by_one_line(line)
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), test_size, train_size + 2 * test_size)):
            train_sample.increment_by_one_line(line)
    with open(data_file_path) as f:
        for line_no, line in enumerate(
                itertools.islice(f.__iter__(), train_size + test_size, train_size + 2 * test_size)):
            part_train_sample.increment_by_one_line(line)
    return test_sample, train_sample, part_train_sample


if __name__ == '__main__':
    smp_1, smp_2, smp_3 = read_sample('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt', 1, 1)
    # print(smp_1.y)
    # print(smp_2.y)
    # print(smp3.y)
    #
    # print(smp_1.element_x)
    # print(smp_2.element_x)
    # print(smp_1.max_feature)
    # print(smp_2.max_feature)
    # print(smp_1.max_class_label)
    # print(smp_2.max_class_label)
    # print(smp_1.row_indptr_x)
    # print(smp_2.row_indptr_x)

    print smp_2.convert_to_csr(1250536 + 1)
    print "======"

    print smp_1.convert_to_csr(1250536 + 1)
    print "======"

    print smp_3.convert_to_csr(1250536 + 1)
    print "======"
