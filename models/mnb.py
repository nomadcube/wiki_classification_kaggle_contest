# coding=utf-8
import math
import numpy as np

from scipy.sparse import csc_matrix
from memory_profiler import profile

from divide_conquer_predict import AllSamplePrediction, OneLabelScore
from model_serialization import save_with_protocol2, load_with_protocol2


class LaplaceSmoothedMNB:
    def __init__(self, model_store_dir):
        self.model_store_dir = model_store_dir
        self._alpha = 1.
        self.num_model = None

    def fit(self, train_y, train_x, part_size, max_y_size):
        b = self.estimate_b(train_y)
        save_with_protocol2(b, self.model_store_dir, 'b.dat')

        all_part_y = self.split(train_y, part_size, max_y_size)
        for j, (part_y, label_list) in enumerate(all_part_y):
            # print "{0} parts have been trained.".format(j)
            part_w = self.estimate_w(part_y, train_x)
            save_with_protocol2(part_w, self.model_store_dir, 'w_{0}.dat'.format(j))
            save_with_protocol2(label_list, self.model_store_dir, 'label_list_{0}.dat'.format(j))
            self.num_model = j + 1

    def predict(self, test_x, predict_cnt):
        """
        :param test_x: csr矩阵
        :param predict_cnt: int
        :return: test_x的各个instance对应的predict_cnt个预测结果
        """
        cnt_instance = test_x.shape[0]
        all_sample_predict = AllSamplePrediction(cnt_instance)
        b = load_with_protocol2(self.model_store_dir, 'b.dat')
        for j in xrange(self.num_model):
            # print "{0} parts have been scored.".format(j)
            part_w = load_with_protocol2(self.model_store_dir, 'w_{0}.dat'.format(j))
            label_list = load_with_protocol2(self.model_store_dir, 'label_list_{0}.dat'.format(j))
            self.label_scoring(b, part_w, all_sample_predict, label_list, test_x, predict_cnt)
        return all_sample_predict.pop_max(predict_cnt)

    def estimate_w(self, part_y, x):
        y_x_param = part_y.dot(x).todense()
        y_x_param += self._alpha
        tmp = np.array(y_x_param.sum(axis=1).ravel())[0]
        y_x_param = y_x_param.transpose()
        y_x_param /= tmp
        y_x_param = np.log(y_x_param)
        return csc_matrix(y_x_param.transpose())

    @staticmethod
    def estimate_b(y):
        each_label_occurrence = np.array(y.sum(axis=1).ravel())[0]
        total_occurrence = each_label_occurrence.sum()
        each_label_occurrence /= total_occurrence
        each_label_occurrence = np.log(each_label_occurrence)
        return each_label_occurrence

    def label_scoring(self, b, part_w, all_part_predict, real_labels, x, k=1):
        log_likelihood_mat = x.dot(part_w.transpose())
        part_b = np.array([b[label] for label in real_labels])
        for i, each_x in enumerate(log_likelihood_mat):
            tmp = np.array(each_x.todense())[0] + part_b
            for j, max_label in enumerate(self.top_k_label(tmp, k)):
                all_part_predict.push(i, OneLabelScore(real_labels[max_label], tmp[max_label]))

    @staticmethod
    def top_k_label(arr, k):
        return np.argsort(arr)[-k:]

    @staticmethod
    def split(whole_y, part_size, max_y_size):
        total_label_list = range(whole_y.shape[0])
        total_size = min(whole_y.shape[0], max_y_size)
        part_cnt = int(math.ceil(float(total_size) / part_size))
        for p in xrange(part_cnt):
            begin = p * part_size
            end = min(total_size, (p + 1) * part_size)
            yield whole_y[begin: end, :], total_label_list[begin: end]
