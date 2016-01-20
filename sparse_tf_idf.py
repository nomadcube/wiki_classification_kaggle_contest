import math
import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix


def counting_occurrence(array_like):
    occurrence = dict()
    for each_element in array_like:
        occurrence.setdefault(each_element, 0)
        occurrence[each_element] += 1
    return occurrence


def tf(count_mat):
    row_sum = csr_matrix.sum(count_mat, axis=1).ravel()
    count_mat.data /= np.array(row_sum.repeat(np.diff(count_mat.indptr)))[0]
    return count_mat


def idf(column, sample_size):
    total_doc_count = sample_size
    feature_occurrence = counting_occurrence(column)
    init_row = list()
    init_element = list()
    for feature, occurrence in feature_occurrence.items():
        init_row.append(feature)
        init_element.append(math.log(float(total_doc_count) / occurrence))
    return csr_matrix((init_element, (init_row, init_row)), shape=(max(init_row) + 1, max(init_row) + 1))


def part_tf_idf_generator(part_csr_dir, global_idf):
    for part_file_name in os.listdir(part_csr_dir):
        with open(os.path.join(part_csr_dir, part_file_name), 'r') as f:
            part_sample = pickle.load(f)
            origin_mat = csr_matrix((part_sample.element_x, (part_sample.row_index_x, part_sample.col_index_x)),
                                    shape=(max(part_sample.row_index_x) + 1, global_idf.shape[0]))
            yield tf(origin_mat).dot(global_idf)


if __name__ == '__main__':
    element = np.array([1., 1., 1., 4., 1., 1.])
    row_index = np.array([0, 1, 1, 1, 2, 2])
    col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
    mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
    tf_mat = tf(mat)
    idf_mat = idf(col_index, 3)
    print(idf_mat)
