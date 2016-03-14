# coding=utf-8
import numpy as np

from array import array
from numpy.ma import masked_values, masked_array
from collections import namedtuple

from transformation.converter import convert_y_to_csr


def evaluation(y, predicted_y):
    evaluation_metrics = namedtuple('evaluation_metrics', 'precision recall f_score accuracy')

    true_positive = 0.
    num_right = 0.
    for i in range(len(y)):
        if y[i] == predicted_y[i]:
            num_right += 1.
            if y[i] == 1.:
                true_positive += 1.

    predict_pos = list(predicted_y).count(1.0)
    y_pos = list(y).count(1.0)

    precision = true_positive / predict_pos
    recall = true_positive / y_pos
    f_score = 1. / (1. / precision + 1. / recall)

    return evaluation_metrics(precision, recall, f_score, num_right / len(y))
