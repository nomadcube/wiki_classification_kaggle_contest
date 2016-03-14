# coding=utf-8
import numpy as np

from copy import deepcopy
from scipy.sparse import csc_matrix
from memory_profiler import profile

from transformation.converter import convert_y_to_csr


class LaplaceSmoothedMNB:
    def __init__(self):
        self.b = None
        self.w = None

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

    @staticmethod
    def estimate_w(y, x):
        label_feature_coef = y.dot(x).todense()
        label_feature_coef += 1.0
        label_sum = np.array(label_feature_coef.sum(axis=1).ravel())[0]
        label_feature_coef = label_feature_coef.transpose()
        label_feature_coef /= label_sum
        label_feature_coef = np.log(label_feature_coef)
        return csc_matrix(label_feature_coef.transpose())

    @staticmethod
    def estimate_b(y):
        each_label_occurrence = np.array(y.sum(axis=1).ravel())[0]
        total_occurrence = each_label_occurrence.sum()
        each_label_occurrence /= total_occurrence
        each_label_occurrence = np.log(each_label_occurrence)
        return each_label_occurrence


class CNB(LaplaceSmoothedMNB):
    def fit(self, train_y, train_x, smp_weight=None):
        y_train_csr = convert_y_to_csr(train_y)
        self.b = self.estimate_b(y_train_csr)
        turned_train_y = list()
        for each_label in train_y:
            new_label = None
            if each_label != 0:
                new_label = 0
            if each_label != 1:
                new_label = 1
            if each_label != 2:
                new_label = 2
            turned_train_y.append(new_label)
        self.w = self.estimate_w(y_train_csr, train_x)

    def post_prob(self, x):
        prob = (-1.) * x.dot(self.w.transpose()).todense() + self.b
        return prob


class NonSmoothedMNB(LaplaceSmoothedMNB):
    def fit(self, train_y, train_x, smp_weight=None):
        weighted_train_x = deepcopy(train_x)
        if smp_weight is not None:
            weighted_train_x.data *= smp_weight.repeat(np.diff(train_x.indptr))
        y_train_csr = convert_y_to_csr(train_y)
        self.b = self.estimate_b(y_train_csr)
        self.w = self.estimate_w(y_train_csr, weighted_train_x)

    @staticmethod
    def estimate_w(y, x):
        label_feature_coef = y.dot(x).todense()
        label_feature_coef += 0.0
        label_sum = np.array(label_feature_coef.sum(axis=1).ravel())[0]
        label_feature_coef = label_feature_coef.transpose()
        label_feature_coef /= label_sum
        label_feature_coef = np.log(label_feature_coef)
        return csc_matrix(label_feature_coef.transpose())
