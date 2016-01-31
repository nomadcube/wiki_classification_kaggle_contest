import array
import itertools


class Sample:
    def __init__(self):
        self.y = list()
        self.element_x = array.array('f')
        self.row_index_x = array.array('I')
        self.col_index_x = array.array('I')
        self.max_feature = -1
        self.max_class_label = -1

    def increment_by_one_line(self, line_no, line):
        multi_label, instance = line.strip().split(' ', 1)
        new_multi_label = array.array('I')
        for label in multi_label.split(','):
            label = int(label)
            new_multi_label.append(label)
            self.max_class_label = label if label > self.max_class_label else self.max_class_label
        self.y.append(new_multi_label)
        for feature in instance.split(' '):
            column, element = feature.split(':')
            column = int(column)
            self.element_x.append(float(element))
            self.col_index_x.append(column)
            self.row_index_x.append(int(line_no))
            self.max_feature = column if column > self.max_feature else self.max_feature
        return self


def read_sample(data_file_path, total_sample_size, train_sample_size):
    train_sample = Sample()
    test_sample = Sample()
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), 0, train_sample_size)):
            train_sample.increment_by_one_line(line_no, line)
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), train_sample_size, total_sample_size)):
            test_sample.increment_by_one_line(line_no, line)
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
