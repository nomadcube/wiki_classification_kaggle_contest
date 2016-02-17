# coding=utf-8
from scipy.optimize import minimize
import numpy as np


def _max_class(y):
    max_class = -1
    for each_y in y:
        for each_label in each_y:
            if each_label > max_class:
                max_class = each_label
    return max_class


def p_norm(w, p):
    return abs(pow(sum(np.power(w, p)), 1. / p))


def discrimination(one_w, one_x):
    return one_w * one_x.transpose()


def empirical_risk(w, y, x):
    res = 0.
    C = _max_class(y)
    w_mat = np.array(w).reshape((C, -1))
    denominator_mat = np.exp(np.dot(w_mat, x.transpose()))
    denominator = denominator_mat.sum(axis=0) + 1.
    for i, each_y in enumerate(y):
        for each_label in each_y:
            if each_label == C:
                current_posterior_prob = 1. / denominator[0, i]
            else:
                current_posterior_prob = denominator_mat[each_label, i] / denominator[0, i]
            res += np.log(current_posterior_prob)
    return (-1.) * res


class LR:
    def __init__(self, regularization_coefficient, p):
        self._regularization_coefficient = regularization_coefficient
        self._p = p
        self.w = None

    def fit(self, y, X):
        print _max_class(y)
        init_w = [[0.1] * X.shape[1] for _ in range(_max_class(y))]
        self.w = minimize(
            lambda w: empirical_risk(w, y, X),
            init_w).x
        self.w = np.array(self.w).reshape((_max_class(y), -1))

    def predict(self, x):
        estimated_y = list()
        for i, each_x in enumerate(x):
            estimated_y.append(self._one_predict(each_x))
        return estimated_y

    def _one_predict(self, one_x):
        all_discrimination_val = np.dot(self.w, one_x.transpose())
        max_discrimination_val = max(all_discrimination_val)
        max_i = np.argmax(all_discrimination_val)
        if max_discrimination_val > 0:
            return [max_i]
        else:
            return [self.w.shape[0]]


if __name__ == '__main__':
    from array import array

    x = [array('f', [2, 3, 3, 3, 2]), array('f', [2, 1, 1, 1, 3]), array('f', [1, 1, 2, 2, 3]),
         array('f', [3, 1, 1, 1, 2]),
         array('f', [2, 2, 3, 3, 2]), array('f', [1, 3, 2, 1, 1]), array('f', [1, 1, 2, 3, 1]),
         array('f', [1, 1, 1, 1, 3]),
         array('f', [3, 2, 1, 1, 3]), array('f', [3, 1, 2, 3, 3])]
    x = np.matrix(np.array(x).reshape((10, 5)))
    print x
    y = [[0], [1], [0], [0], [0], [1], [0], [0], [1], [1]]
    print y
    lr = LR(0.3, 1)
    print _max_class(y)
    lr.fit(y, x)
    print lr.w
    print lr.predict(x)
    #
    # print p_norm([1, -2], 1)
    # print p_norm([1, -2], 2)
