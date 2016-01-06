from collections import namedtuple
from random import random

from scipy.sparse import csr_matrix

from data.tf_idf import x_with_tf_idf


def base_sample_reader(data_file_path, sample_prop=1.0):
    """
    Read data from data_file_path and convert it to a Sample object.

    :param data_file_path: str
    :param sample_prop: float
    :return: dict
    """
    sample = namedtuple('sample', 'y x')
    y = list()
    x = list()
    with open(data_file_path, 'r') as f_stream:
        line = f_stream.readline()
        while line:
            determine_number = random()
            if determine_number <= sample_prop:
                tmp_y, tmp_x = line.strip().split(' ', 1)
                y.append(tmp_y)
                x.append({column.split(':')[0]: float(column.split(':')[1]) for column in tmp_x.split(' ')})
            line = f_stream.readline()
    return sample(y, x)


def dimension_reduction(x, threshold):
    """
    Use tf-idf to perform dimension reduction, update x and y.

    :param x: namedtuple
    :param threshold: float
    :return: Sample
    """
    return x_with_tf_idf(x, threshold)


def construct_csr_sample(base_x, feature_mapping_relation=None):
    """
    Convert to csr format with feature remapped to col index.

    :param base_x: list
    :param feature_mapping_relation: {None, dict}
    :return csr_matrix
    """
    if not feature_mapping_relation:
        data = list()
        row_ind = list()
        col_ind = list()
        feature_mapping = dict()
        for instance_id, instance in enumerate(base_x):
            if len(instance) == 0:
                continue
            for feature, feature_val in instance.items():
                row_ind.append(instance_id)
                feature_num = len(feature_mapping)
                feature_mapping.setdefault(feature, feature_num)
                col_ind.append(feature_mapping[feature])
                data.append(feature_val)
        return csr_matrix((data, (row_ind, col_ind)), shape=(len(base_x), len(feature_mapping))), feature_mapping
    else:
        data = list()
        row_ind = list()
        col_ind = list()
        for instance_id, instance in enumerate(base_x):
            if len(instance) == 0:
                continue
            for feature in feature_mapping_relation.keys():
                row_ind.append(instance_id)
                col_ind.append(feature_mapping_relation[feature])
                val = instance[feature] if instance.get(feature) is not None else 0.0
                data.append(val)
        return csr_matrix((data, (row_ind, col_ind)), shape=(len(base_x), max(col_ind) + 1))


if __name__ == '__main__':
    print(construct_csr_sample([{'1250536': 1},
                                {},
                                {'1250536': 1, '805104': 1}]))
