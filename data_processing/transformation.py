import linecache
from collections import namedtuple
from random import random, randrange

from scipy.sparse import csr_matrix

from data_processing.tf_idf import x_with_tf_idf

from memory_profiler import profile


def sample_line_num(max_line_num, sample_size):
    res = set()
    while len(res) < sample_size:
        new_line_num = randrange(1, max_line_num)
        if new_line_num not in res:
            res.add(new_line_num)
    return res


def sample_reader(data_file_path, max_line_num, sample_size):
    """
    Read data_processing from data_file_path and convert it to a Sample object.

    :param data_file_path: str
    :param max_line_num: int
    :param sample_size: int
    :return: sample
    """
    sample = namedtuple('sample', 'y x')
    y = list()
    x = list()
    selected_line_no = sample_line_num(max_line_num, sample_size)
    for line_no in selected_line_no:
        line = linecache.getline(data_file_path, line_no)
        tmp_y, tmp_x = line.strip().split(' ', 1)
        y.append([int(tmp_label) for tmp_label in tmp_y.split(',')])
        instance = dict()
        for column in tmp_x.split(' '):
            k, v = column.split(':')
            instance[int(k)] = int(v)
        x.append(instance)
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


def flatting_multi_label(original_y, original_x):
    if len(original_x) != len(original_y):
        raise ValueError('Length of original_y and original_y must be the same.')
    new_y = list()
    new_x = list()
    flatting_rel = dict()
    new_instance_id = 0
    for instance_id in range(len(original_y)):
        flatting_rel.setdefault(instance_id, set())
        each_original_y = original_y[instance_id]
        each_original_x = original_x[instance_id]
        labels = original_y[instance_id].split(',')
        if len(labels) <= 1:
            new_y.append(each_original_y)
            new_x.append(each_original_x)
            flatting_rel[instance_id].add(new_instance_id)
            new_instance_id += 1
        else:
            for each_label in labels:
                new_y.append(each_label)
                new_x.append(each_original_x)
                flatting_rel[instance_id].add(new_instance_id)
                new_instance_id += 1
    return new_y, new_x, flatting_rel


def assemble_y(flat_y, flat_relation):
    new_y = list()
    for y_collection_index in flat_relation.values():
        tmp_y = set()
        for i in y_collection_index:
            tmp_y.add(flat_y[i])
        new_y.append(','.join(tmp_y))
    return new_y


if __name__ == '__main__':
    print(construct_csr_sample([{'1250536': 1},
                                {},
                                {'1250536': 1, '805104': 1}]))
    import numpy as np

    X = np.random.random_sample([2, 10])
    np.dot(X, X.T)
