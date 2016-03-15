# coding=utf-8
import numpy as np

from copy import deepcopy
from scipy.sparse import csc_matrix, csr_matrix
from memory_profiler import profile

from base import normalized_by_row_sum
from transformation.converter import convert_y_to_csr


class LaplaceSmoothedMNB:
    def __init__(self, alpha=1.0):
        self.b = None
        self.w = None
        self._alpha = alpha

    def fit(self, train_y, train_x, smp_weight=None):
        y_train_csr = convert_y_to_csr(train_y)
        self.b = self.estimate_b(y_train_csr)
        self.w = self.estimate_w(y_train_csr, train_x)

    def predict(self, x):
        prob = self.post_prob(x)
        return np.array(np.argmax(prob, axis=1).ravel())[0]

    def post_prob(self, x):
        prob = x.dot(self.w.transpose()).todense() + self.b
        return prob

    def estimate_w(self, y, x):
        label_feature_coef = y.dot(x).todense()
        label_feature_coef += self._alpha
        label_feature_coef = normalized_by_row_sum(label_feature_coef)
        label_feature_coef = np.log(label_feature_coef)
        return csc_matrix(label_feature_coef.transpose())

    @staticmethod
    def estimate_b(y):
        each_label_occurrence = np.array(y.sum(axis=1).ravel())[0]
        each_label_occurrence = normalized_by_row_sum(each_label_occurrence)
        return each_label_occurrence


class CNB(LaplaceSmoothedMNB):
    def post_prob(self, x):
        prob = (-1.) * x.dot(self.w.transpose()).todense() + self.b
        return prob

    def estimate_w(self, y, x):
        new_y = y.todense()
        new_y = csr_matrix(1.0 - new_y)
        label_feature_coef = new_y.dot(x).todense()
        label_feature_coef += self._alpha
        label_feature_coef = normalized_by_row_sum(label_feature_coef)
        label_feature_coef = np.log(label_feature_coef)
        return csc_matrix(label_feature_coef.transpose())


class WeightedNonSmoothedMNB(LaplaceSmoothedMNB):
    def __init__(self, alpha):
        LaplaceSmoothedMNB.__init__(self, alpha=alpha)

    def fit(self, train_y, train_x, smp_weight=None):
        weighted_train_x = deepcopy(train_x)
        if smp_weight is not None:
            weighted_train_x.data *= smp_weight.repeat(np.diff(train_x.indptr))
        y_train_csr = convert_y_to_csr(train_y)
        self.b = self.estimate_b(y_train_csr)
        self.w = self.estimate_w(y_train_csr, weighted_train_x)
