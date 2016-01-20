import itertools
import os
import pickle


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
            self.element_x.append(float(element))
            self.col_index_x.append(int(column))
            self.row_index_x.append(int(line_no))
        return self


def read_part_sample(data_file_path, total_sample_size, part_one_sample_size):
    part_one_sample = Sample()
    part_two_sample = Sample()
    with open(data_file_path) as f:
        for line_no, line in enumerate(itertools.islice(f.__iter__(), total_sample_size)):
            if line_no <= part_one_sample_size:
                part_one_sample.increment_by_one_line(line_no, line)
            else:
                part_two_sample.increment_by_one_line(line_no, line)
    return part_one_sample, part_two_sample


def save_part_sample(part_sample, part_sample_dir):
    for part_index, part in enumerate(part_sample):
        with open(os.path.join(part_sample_dir, 'sample_{0}.obj'.format(part_index)), 'w') as f:
            pickle.dump(part, f)


if __name__ == '__main__':
    smp_1, smp_2 = read_part_sample('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt', 3, 1)
    print(smp_1.y)
    print(smp_2.y)
    print(smp_1.element_x)
    print(smp_2.element_x)
