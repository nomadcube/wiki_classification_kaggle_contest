from collections import namedtuple

import numpy as np
import numpy.ma as nma

from preprocessing import convert_y_to_csr


def confusion_matrix(y, predicted_y, max_n_dim):
    """
    :param y: list
    :param predicted_y: list
    :param max_n_dim: int
    :return: confusion_matrix
    """
    res = namedtuple('confusion_matrix', 'true_positive false_negative false_positive true_negative')

    y_mat = convert_y_to_csr(y, element_dtype='bool', max_n_dim=max_n_dim).transpose()
    pred_mat = convert_y_to_csr(predicted_y, element_dtype='bool', max_n_dim=max_n_dim).transpose()

    y_pred_logical_and = y_mat.multiply(pred_mat)

    y_pred_positive = np.array(y_pred_logical_and.sum(axis=1).ravel(), dtype='int32')[0]
    y_positive = np.array(y_mat.sum(axis=1).ravel(), dtype='int32')[0]
    pred_positive = np.array(pred_mat.sum(axis=1).ravel(), dtype='int32')[0]
    return res(y_pred_positive, (y_positive - y_pred_positive), (pred_positive - y_pred_positive),
               (y_mat.shape[0] - (y_positive + (pred_positive - y_pred_positive))))


def macro_precision_recall(y, predicted_y, max_n_dim):
    confusion_mat = confusion_matrix(y, predicted_y, max_n_dim)
    y_positive = nma.masked_values(confusion_mat.true_positive + confusion_mat.false_negative, 0.)
    pred_positive = nma.masked_values(confusion_mat.true_positive + confusion_mat.false_positive, 0.)
    precision = nma.masked_array(confusion_mat.true_positive, y_positive.mask) / y_positive
    recall = nma.masked_array(confusion_mat.true_positive, pred_positive.mask) / pred_positive
    return precision.mean(), recall.mean()


if __name__ == '__main__':
    test_y = [[314523, 165538, 416827], [21631], [76255, 335416]]
    test_predicted_y = [[165538], [416827], [76255, 335416]]
    print('c_11')
    print(confusion_matrix(test_y, test_predicted_y, 416828)[0])
    print('c_10')
    print(confusion_matrix(test_y, test_predicted_y, 416828)[1])
    print('c_01')
    print(confusion_matrix(test_y, test_predicted_y, 416828)[2])
    print('c_00')
    # print(confusion_matrix(test_y, test_predicted_y)[3])
    # print(precision_recall(test_y, test_predicted_y)[0])
    # print(precision_recall(test_y, test_predicted_y)[1])
    print(macro_precision_recall(test_y, test_predicted_y, 416828)[0])
    print(macro_precision_recall(test_y, test_predicted_y, 416828)[1])
