import array
import itertools


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

        all_labels = array.array('I', [int(l) for l in multi_label.split(',')])
        self.y.append(all_labels)
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


def read_sample(data_file_path, total_sample_size, train_sample_size):
    train_sample = Sample()
    test_sample = Sample()
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), 0, train_sample_size)):
            train_sample.increment_by_one_line(line)
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), train_sample_size, total_sample_size)):
            test_sample.increment_by_one_line(line)
    return train_sample, test_sample


if __name__ == '__main__':
    smp_1, smp_2 = read_sample('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt', 3, 1)
    print(smp_1.y)
    print(smp_2.y)
    print(smp_1.element_x)
    print(smp_2.element_x)
    print(smp_1.max_feature)
    print(smp_2.max_feature)
    print(smp_1.max_class_label)
    print(smp_2.max_class_label)
    print(smp_1.row_indptr_x)
    print(smp_2.row_indptr_x)
