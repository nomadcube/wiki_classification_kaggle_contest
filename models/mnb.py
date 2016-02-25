# coding=utf-8
import numpy as np
from numpy.ma import masked_values
from scipy.sparse import csr_matrix
from abc import abstractmethod
from memory_profiler import profile
import math
import heapq
from pickle import dump, load


class OnePrediction(object):
    __slots__ = ('label', 'score')

    def __init__(self, label, score):
        self.label = label
        self.score = score

    def __le__(self, other):
        # todo: 符号不一致因为heapq默认是最小堆
        return self.score > other.score


class BaseMNB:
    def __init__(self, model_store_dir):
        self.model_store_dir = model_store_dir
        self.num_model = None

    def fit(self, train_y, train_x, part_size):
        b = self._estimate_b(train_y)
        with open('{0}/b.dat'.format(self.model_store_dir), 'wb') as b_f:
            dump(b, b_f)
        for j, (part_y, label_list) in enumerate(self._y_split(train_y, part_size)):
            print "{0} parts have been trained.".format(j)
            part_w = self._part_estimate_w(part_y, train_x)
            with open('{0}/w_{1}.dat'.format(self.model_store_dir, j), 'wb') as w_f:
                dump(part_w, w_f)
            with open('{0}/label_list_{1}.dat'.format(self.model_store_dir, j), 'wb') as label_list_f:
                dump(label_list, label_list_f)
            self.num_model = j

    def predict(self, test_x, predict_cnt):
        cnt_instance = test_x.shape[0]
        all_part_predict = [[] for _ in range(cnt_instance)]
        with open('{0}/b.dat'.format(self.model_store_dir), 'r') as b_f:
            b = load(b_f)
        for j in xrange(self.num_model):
            print "{0} parts have been scored.".format(j)
            with open('{0}/w_{1}.dat'.format(self.model_store_dir, j), 'r') as w_f:
                part_w = load(w_f)
            with open('{0}/label_list_{1}.dat'.format(self.model_store_dir, j), 'r') as label_list_f:
                label_list = load(label_list_f)
            self._part_scoring(b, part_w, all_part_predict, label_list, test_x, predict_cnt)
        return [[heapq.heappop(part_pred).label for _ in range(min(predict_cnt, len(part_pred)))] for part_pred in
                all_part_predict]

    @staticmethod
    def _estimate_b(y):
        y_col_sum = np.array(y.sum(axis=0))[0]
        total_label_occurrence_cnt = y_col_sum.sum()
        y_col_sum /= total_label_occurrence_cnt
        y_col_sum = masked_values(y_col_sum, 0.)
        y_col_sum = np.log(y_col_sum)
        return y_col_sum

    @staticmethod
    def _y_split(whole_y, part_size):
        total_label_list = np.unique(whole_y.indices)
        lil_y = whole_y.transpose().tolil()
        total_size = lil_y.shape[0]
        part_cnt = int(math.ceil(float(total_size) / part_size))
        for p in xrange(part_cnt):
            begin = p * part_size
            end = min(total_size, (p + 1) * part_size)
            yield lil_y[begin: end].tocsr().transpose(), total_label_list[begin: end]

    @abstractmethod
    def _part_estimate_w(self, part_y, x):
        pass

    @abstractmethod
    def _part_scoring(self, b, w, all_part_predict, real_labels, x):
        pass


class LaplaceSmoothedMNB(BaseMNB):
    def __init__(self, model_store_dir):
        BaseMNB.__init__(self, model_store_dir)
        self._alpha = 1.

    # @profile
    def _part_estimate_w(self, part_y, x):
        y_x_param = part_y.transpose().dot(x)
        y_x_param = y_x_param.todense()
        y_x_param += self._alpha
        tmp = np.array(y_x_param.sum(axis=1).ravel())[0]
        y_x_param = y_x_param.transpose()
        y_x_param /= tmp
        y_x_param = np.log(y_x_param)
        return csr_matrix(y_x_param.transpose())

    # @profile
    def _part_scoring(self, b, part_w, all_part_predict, real_labels, x, k=1):
        log_likelihood_mat = x.dot(part_w.transpose())
        part_b = np.array([b[label] for label in real_labels])
        for i, each_x in enumerate(log_likelihood_mat):
            tmp = np.array(each_x.todense())[0] + part_b
            for j, max_label in enumerate(self._top_k_argmax(tmp, k)):
                all_part_predict[i].append(OnePrediction(real_labels[max_label], tmp[max_label]))

    @staticmethod
    def _top_k_argmax(arr, k):
        return np.argsort(arr)[-k:]


if __name__ == '__main__':
    # from preprocessing import transforming
    #
    # y = transforming.convert_y_to_csr(np.array([[0], [0], [1], [1], [0],
    #                                             [0], [0], [1], [1], [1],
    #                                             [1], [1], [1], [1], [0]]))
    # element = [1.] * 30
    # row_index = list()
    # for i in range(15):
    #     row_index.extend([i] * 2)
    # col_index = [0, 3,
    #              0, 4,
    #              0, 4,
    #              0, 3,
    #              0, 3,
    #              1, 3,
    #              1, 4,
    #              1, 4,
    #              1, 5,
    #              1, 5,
    #              2, 5,
    #              2, 4,
    #              2, 4,
    #              2, 5,
    #              2, 5]
    # x = csr_matrix((element, (row_index, col_index)), shape=(15, 6))
    # m = LaplaceSmoothedMNB()
    # print m.fit(y, x, x, 2, 1)
    # print m.fit(y, x, x, 1, 1)
    arr = np.array([2, 3, 5, 1])
    print _top_k_argmax(arr, 2)
