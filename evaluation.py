from collections import namedtuple

import numpy as np
import numpy.ma as nma

from fit_multi_label_mnb import construct_csr_from_list


def confusion_matrix(y, predicted_y, max_n_dim):
    res = namedtuple('confusion_matrix', 'true_positive false_negative false_positive true_negative')
    y_coo = construct_csr_from_list(y, max_n_dim).transpose()
    predicted_y_coo = construct_csr_from_list(predicted_y, max_n_dim).transpose()
    y_csr = y_coo.tocsr()
    predicted_y_csr = predicted_y_coo.tocsr()
    tp_mat = y_csr.multiply(predicted_y_csr)
    c_11 = tp_mat.sum(axis=1)
    c_10_11 = y_csr.sum(axis=1)
    c_01_11 = predicted_y_csr.sum(axis=1)
    c_01_10_00_11 = np.matrix([y_csr.shape[1]] * y_csr.shape[0]).reshape((y_csr.shape[0], 1))
    return res(c_11, (c_10_11 - c_11), (c_01_11 - c_11), (c_01_10_00_11 - (c_10_11 + (c_01_11 - c_11))))


def precision_recall(y, predicted_y, max_n_dim):
    confusion_mat = confusion_matrix(y, predicted_y, max_n_dim)
    flat_arr = np.array(confusion_mat.true_positive + confusion_mat.false_negative).flatten()
    all_true = nma.masked_values(flat_arr, 0.0)
    all_positive = nma.masked_values((confusion_mat.true_positive + confusion_mat.false_positive).flatten(), 0.0)
    precision = nma.masked_array(np.array(confusion_mat.true_positive).flatten(), all_true.mask) / all_true
    recall = nma.masked_array(np.array(confusion_mat.true_positive).flatten(), all_positive.mask) / all_positive
    return precision, recall


def macro_precision_recall(y, predicted_y, max_n_dim=None):
    pre, rec = precision_recall(y, predicted_y, max_n_dim)
    return pre.mean(), rec.mean()


if __name__ == '__main__':
    test_y = [[314523, 165538, 416827], [21631], [76255, 335416]]
    test_predicted_y = [[165538], [416827], [76255, 335416]]
    print('c_11')
    print(confusion_matrix(test_y, test_predicted_y)[0])
    print('c_10')
    print(confusion_matrix(test_y, test_predicted_y)[1])
    print('c_01')
    print(confusion_matrix(test_y, test_predicted_y)[2])
    print('c_00')
    # print(confusion_matrix(test_y, test_predicted_y)[3])
    # print(precision_recall(test_y, test_predicted_y)[0])
    # print(precision_recall(test_y, test_predicted_y)[1])
    print(macro_precision_recall(test_y, test_predicted_y)[0])
    print(macro_precision_recall(test_y, test_predicted_y)[1])
