import itertools


class Sample:
    def __init__(self):
        self.y = list()
        self.element_x = list()
        self.row_index_x = list()
        self.col_index_x = list()

    def increment_by_one_line(self, line_no, line):
        multi_label, instance = line.strip().split(' ', 1)
        self.y.append([int(label) for label in multi_label.split(',')])
        for feature in instance.split(' '):
            column, element = feature.split(':')
            self.element_x.append(int(element))
            self.col_index_x.append(int(column))
            self.row_index_x.append(int(line_no))
        return self


def sample_reader(data_file_path, begin_line_no, split_line_no, end_line_no):
    part_one_sample = Sample()
    part_two_sample = Sample()
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), begin_line_no, split_line_no)):
            part_one_sample.increment_by_one_line(line_no, line)
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), split_line_no, end_line_no)):
            part_two_sample.increment_by_one_line(line_no, line)
    return part_one_sample, part_two_sample


if __name__ == '__main__':
    smp_1, smp_2 = sample_reader('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt', 0, 2, 3)
    print(smp_1.y)
    print(smp_2.y)

    print(smp_1.element_x)
    print(smp_2.element_x)
