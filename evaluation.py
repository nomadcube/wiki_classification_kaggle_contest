# coding=utf-8
import numpy as np

from array import array
from numpy.ma import masked_values, masked_array
from collections import namedtuple

from transformation.converter import convert_y_to_csr


def evaluation(y, predicted_y):
    evaluation_metrics = namedtuple('evaluation_metrics', 'mpr mre f_score accuracy')

    y_mat = convert_y_to_csr(y)
    pred_mat = convert_y_to_csr(predicted_y)

    non_empty_rows_no = np.diff(y_mat.indptr)
    num_y_label = np.where(non_empty_rows_no > 0)[0].shape[0]

    inter_mat = y_mat.multiply(pred_mat)
    true_positive = inter_mat.sum(axis=1)
    num_right = inter_mat.sum()

    # 将不在y_mat中出现的label行都masked掉，此逻辑同时作用于真实的y和对y的预测值
    y_pos = masked_values(y_mat.sum(axis=1), 0.)
    pred_pos = masked_array(pred_mat.sum(axis=1), y_pos.mask)

    precision = true_positive / y_pos
    recall = true_positive / pred_pos
    metrics_denominator = num_y_label
    m_precision = precision.sum() / metrics_denominator
    m_recall = recall.sum() / metrics_denominator
    f_score = 2. / (1. / m_precision + 1. / m_recall) if m_precision != 0. and m_recall != 0. else float("inf")

    return evaluation_metrics(m_precision, m_recall, f_score, num_right / len(y))
