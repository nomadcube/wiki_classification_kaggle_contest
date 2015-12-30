# coding=utf-8
from tf_idf_swig.tf_idf import tf_idf, doc_term_val_t
from collections import namedtuple
import io


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

    def label_string_disassemble(self):
        """
        Disassemble compounded label string to single label,
        while the corresponding x becomes duplicated.
        """
        new_y = dict()
        new_x = dict()
        new_key = 0
        for k in self.y.keys():
            original_x = self.x[k]
            for single_label in self.y[k].split(','):
                self.index_mapping_relation[new_key] = k
                new_y[new_key] = single_label
                new_x[new_key] = original_x
                new_key += 1
        self.y = new_y
        self.x = new_x
        return self

    def dimension_reduction(self, threshold):
        """
        Use tf-idf to perform dimension reduction, update x and y.

        :param threshold: float
        :return: Sample
        """
        self.x = dict(tf_idf(doc_term_val_t(self.x), threshold))
        for k in self.x:
            self.x[k] = dict(self.x[k])
        self.y = {k: v for (k, v) in self.y.items() if k in self.x.keys()}
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

    def feature_dimension(self):
        feature_set = set()
        for instance_index in self.x.keys():
            for feature in self.x[instance_index].keys():
                feature_set.add(feature)
        return len(feature_set)

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
            converted_y = 1 if positive_flag == str(self.y[y_key]) else -1
            self.binary_y[y_key] = converted_y
        return self


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
