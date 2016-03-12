# coding=utf-8
import numpy as np

from scipy.sparse import csc_matrix
from memory_profiler import profile

from preprocessing.transforming import convert_y_to_csr


class LaplaceSmoothedMNB:
    def __init__(self):
        self.b = None
        self.w = None

    def fit(self, train_y, train_x):
        y_train_csr = convert_y_to_csr(train_y)
        self.b = self.estimate_b(y_train_csr)
        self.w = self.estimate_w(y_train_csr, train_x)

    def predict(self, test_x, predict_cnt=1):
        """
        :param test_x: csr矩阵
        :param predict_cnt: int
        :return: test_x的各个instance对应的predict_cnt个预测结果

        当test_x为展开label的状态，即如314523,165538,416827 1250536:1这一行会展开为3个相同的instance， 这3个instance的预测结果是一样的
        这并不影响模型评估指标的计算，因为这相当于将预测结果按instance group by 之后，以其中的一个预测结果作为group by 后的预测结果
        """
        prediction = list()
        log_likelihood_mat = test_x.dot(self.w.transpose())
        for i, each_x in enumerate(log_likelihood_mat):
            post_probability = (-1.) * np.array(each_x.todense())[0] + self.b
            prediction.append([label for label in self.top_k_label(post_probability, predict_cnt)])
        return prediction

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

    @staticmethod
    def estimate_b(y):
        each_label_occurrence = np.array(y.sum(axis=1).ravel())[0]
        total_occurrence = each_label_occurrence.sum()
        each_label_occurrence /= total_occurrence
        each_label_occurrence = np.log(each_label_occurrence)
        return each_label_occurrence

    @staticmethod
    def top_k_label(arr, k):
        return np.argsort(arr)[-k:]
