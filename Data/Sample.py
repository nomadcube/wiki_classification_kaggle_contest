import io
from collections import namedtuple

from Data.tf_idf.tf_idf import tf_idf


class Sample:
    def __init__(self):
        self.y = dict()
        self.x = dict()
        self.binary_y = dict()
        self.index_mapping_relation = dict()

    def size(self):
        if len(self.y) != len(self.x):
            raise ValueError('The size of y and x must agree.')
        return len(self.y)

    def dimension_reduction(self, threshold):
        """
        Use tf-idf to perform dimension reduction, update x and y.

        :param threshold: float
        :return: Sample
        """
        self.x = tf_idf(self.x, threshold)
        new_y = dict()
        for k in self.x:
            new_y[k] = self.y[k]
        self.y = new_y
        return self

    def feature_dimension(self):
        feature_set = set()
        for instance_index in self.x.keys():
            for feature in self.x[instance_index].keys():
                feature_set.add(feature)
        return len(feature_set)

    def label_upward(self, upward_hierarchy):
        for y_key in self.y.keys():
            new_labels = list()
            for each_label in self.y[y_key].split(','):
                try:
                    new_each_label = str(upward_hierarchy.dat[int(each_label)].parent_id)
                except IndexError:
                    new_each_label = each_label
                new_labels.append(new_each_label)
            self.y[y_key] = ','.join(new_labels)
        return self

    def description(self):
        sample_desc = namedtuple("DataDesc", "sample_size feature_dimension class_number")
        return sample_desc(self.size(), self.feature_dimension(), len(set(self.y.values())))

    def convert_to_binary_class(self, positive_flag):
        """
        Convert y from multi-class to binary-class,
        if one label contains ```positive_class``` then it would be converted to 1, -1 otherwise.
        :param positive_flag: str
        :return: Sample
        """
        for y_key in self.y.keys():
            converted_y = 1 if positive_flag in self.y[y_key] else -1
            self.binary_y[y_key] = converted_y
        return self

    def split_train_test(self, train_proportion):
        """
        Generate train and test sample from total Sample.

        :param train_proportion: float
        :return: tuple
        """
        train_count = int(self.size() * train_proportion)
        train_keys = self.y.keys()[:train_count]
        test_keys = self.y.keys()[train_count:]
        return [self.binary_y[i] for i in train_keys], \
               [self.x[i] for i in train_keys], \
               [self.binary_y[j] for j in test_keys], \
               [self.x[j] for j in test_keys], train_keys, test_keys


def sample_reader(data_file_path):
    """
    Read data from data_file_path and convert it to a Sample object.

    :param data_file_path: str
    :return: Sample
    """
    sample = Sample()
    for index, line in enumerate(io.open(data_file_path, 'r').readlines()):
        sample.x[index] = dict()
        tmp_y, tmp_x = line.strip().split(' ', 1)
        sample.y[index] = tmp_y
        for column in tmp_x.split(' '):
            feat, val = column.split(':')
            feat = int(feat)
            val = float(val)
            sample.x[index][feat] = val
    return sample


if __name__ == '__main__':
    TR = sample_reader('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')

