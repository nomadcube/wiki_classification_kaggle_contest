from random import random

from scipy.sparse import csr_matrix

from Sample import Sample


def sample_reader(data_file_path, sample_prop=1.0):
    """
    Read data from data_file_path and convert it to a Sample object.

    :param data_file_path: str
    :param sample_prop: float
    :return: Sample
    """
    sample = Sample()
    index = 0
    with open(data_file_path, 'r') as f_stream:
        line = f_stream.readline()
        while line:
            determine_number = random()
            if determine_number <= sample_prop:
                sample.x[index] = dict()
                tmp_y, tmp_x = line.strip().split(' ', 1)
                sample.y[index] = tmp_y
                for column in tmp_x.split(' '):
                    feat, val = column.split(':')
                    feat = int(feat)
                    val = float(val)
                    sample.x[index][feat] = val
            line = f_stream.readline()
            index += 1
    return sample


def construct_csr(row_index, col_index_and_value, constraint_features=None):
    """
    Construct a csr data_set with given key as row_index and value's key as col_index.

    :param row_index: list<numeric>
    :param col_index_and_value: list<dict>
    :param constraint_features: {set<numeric>, None}
    :return csr_matrix
    """
    if constraint_features is None:
        data = list()
        row_ind = list()
        col_ind = list()
        col_num = 0
        for i in range(len(row_index)):
            for feature_index, feature_val in col_index_and_value[i].items():
                data.append(feature_val)
                row_ind.append(i)
                col_ind.append(feature_index)
                if feature_index > col_num:
                    col_num = feature_index
        return csr_matrix((data, (row_ind, col_ind)), shape=(len(row_index), col_num + 1))
    else:
        data = list()
        row_ind = list()
        col_ind = list()
        for i in range(len(row_index)):
            for each_constraint_feature in constraint_features:
                feat_val = col_index_and_value[i][each_constraint_feature] if each_constraint_feature in \
                                                                              col_index_and_value[i].keys() else 0.0
                data.append(feat_val)
                row_ind.append(i)
                col_ind.append(each_constraint_feature)
        return csr_matrix((data, (row_ind, col_ind)), shape=(len(row_index), max(constraint_features) + 1))
