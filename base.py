# coding=utf-8
import numpy as np
from scipy.sparse import csr_matrix


def normalized_by_row_sum(mat):
    """
    将矩阵的元素按各行元素和做标准化，返回一个新矩阵。

    :param mat:{np.matrix, sparse matrix}
    :return: {np.matrix, sparse matrix}
    """
    major_axis = 1 if len(mat.shape) == 2 else 0
    row_sum = np.array(mat.sum(axis=major_axis).ravel())[0]
    if isinstance(mat, np.ndarray):
        normalized_mat = mat.transpose()
        normalized_mat /= row_sum
        return normalized_mat
    if isinstance(mat, csr_matrix):
        normalized_mat = csr_matrix(mat.shape)
        row_sum = csr_matrix.sum(mat, axis=1).ravel()
        normalized_mat.data = mat.data / np.array(row_sum.repeat(np.diff(mat.indptr)))[0]
        normalized_mat.indptr = mat.indptr
        normalized_mat.indices = mat.indices
        return normalized_mat
