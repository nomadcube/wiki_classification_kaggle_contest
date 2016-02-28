# coding=utf-8
import numpy as np

from array import array
from numpy.ma import masked_values
from collections import namedtuple

from preprocessing.transforming import convert_y_to_csr


def get_evaluation_metrics(y, predicted_y):
    # 若某个类别只在y_mat中出现或只在pred_mat中出现，对应的precision和recall都设置成0
    evaluation_metrics = namedtuple('evaluation_metrics', 'mpr mre f_score')
    y_mat = convert_y_to_csr(y, element_dtype='float')
    non_empty_rows_no = np.diff(y_mat.indptr)
    num_y_label = np.where(non_empty_rows_no > 0)[0].shape[0]
    pred_mat = convert_y_to_csr(predicted_y, element_dtype='float', total_label_cnt=y_mat.shape[0])

    inter_mat = y_mat.multiply(pred_mat)
    y_pos = masked_values(y_mat.sum(axis=1), 0.)
    pred_pos = masked_values(y_mat.sum(axis=1), 0.)
    true_positive = inter_mat.sum(axis=1)
    precision = true_positive / y_pos
    recall = true_positive / pred_pos
    m_precision = precision.sum() / num_y_label
    m_recall = recall.sum() / num_y_label
    f_score = 1. / (1. / m_precision + 1. / m_recall) if m_precision != 0. and m_recall != 0. else float("inf")
    return evaluation_metrics(m_precision, m_recall, f_score)


if __name__ == '__main__':
    test_y = [array('I', [65L, 66L, 67L, 68L, 69L]), array('I', [15L, 66L, 17L, 18L])]
    test_predicted_y = [[0], [15]]
    print(get_evaluation_metrics(test_y, test_predicted_y, 70, 8)[0])
    print(get_evaluation_metrics(test_y, test_predicted_y, 70, 8)[1])
