# coding=utf-8
import numpy as np

from scipy.sparse import csc_matrix
from memory_profiler import profile

from transformation.converter import convert_y_to_csr


class LaplaceSmoothedMNB:
    def __init__(self):
        self.b = None
        self.w = None

    def fit(self, train_y, train_x):
        y_train_csr = convert_y_to_csr(train_y)
        self.b = self.estimate_b(y_train_csr)
        self.w = self.estimate_w(y_train_csr, train_x)

    def predict(self, x):
        prob = self.post_prob(x)
        return [[pred] for pred in np.argmax(prob, axis=1)]

    def post_prob(self, x):
        prob = x.dot(self.w.transpose())
        return self.b.take(np.diff(prob.indices)) + prob

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
    @staticmethod
    def estimate_w(y, x):
        y_x_param = y.dot(x).todense()
        y_x_param += 1.0
        tmp = np.array(y_x_param.sum(axis=1).ravel())[0]
        y_x_param = y_x_param.transpose()
        y_x_param /= tmp
        y_x_param = np.log(y_x_param)
        contrast_y_x_param = np.array(
            zip(list(np.array(y_x_param[:, 1]).ravel()), list(np.array(y_x_param[:, 0]).ravel()))).reshape(
            y_x_param.shape)
        return csc_matrix(contrast_y_x_param.transpose())

    def post_prob(self, x):
        prob = x.dot(self.w.transpose())
        return self.b.take(np.diff(prob.indices)) + (-1.) * prob
