# coding=utf-8
import numpy as np
from numpy.ma import masked_values
from scipy.sparse import csr_matrix, lil_matrix
from abc import abstractmethod
from memory_profiler import profile
import math
from itertools import product
import heapq


class OnePrediction:
    def __init__(self, label, score):
        self.label = label
        self.score = score

    def __le__(self, other):
        return self.score < other.score


class BaseMNB:
    def __init__(self):
        self.part_w = None
        self.b = None

    def fit_and_predict(self, train_y, train_x, test_x, part_size, predict_cnt):
        cnt_instance = test_x.shape[0]
        all_part_predict = [[] for _ in range(cnt_instance)]
        self.b = self._estimate_b(train_y)
        for i, (part_y, real_labels) in enumerate(self._y_split(train_y, part_size)):
            self.part_w = self._part_estimate_w(part_y, train_x)
            self._part_scoring(all_part_predict, real_labels, test_x, predict_cnt)
        return [[heapq.heappop(part_pred).label for _ in range(2)] for part_pred in all_part_predict]

    @staticmethod
    def _estimate_b(y):
        y_col_sum = np.array(y.sum(axis=0))[0]
        total_label_occurrence_cnt = y_col_sum.sum()
        y_col_sum /= total_label_occurrence_cnt
        y_col_sum = masked_values(y_col_sum, 0.)
        y_col_sum = np.log(y_col_sum)
        return y_col_sum

    @abstractmethod
    def _part_estimate_w(self, part_y, x):
        pass

    @abstractmethod
    def _part_scoring(self, all_part_predict, real_labels, x):
        pass

    @staticmethod
    def _y_split(whole_y, part_size):
        lil_y = whole_y.transpose().tolil()
        total_size = lil_y.shape[0]
        part_cnt = int(math.ceil(float(total_size) / part_size))
        for p in xrange(part_cnt):
            begin = p * part_size
            end = min(total_size, (p + 1) * part_size)
            yield lil_y[begin: end].tocsr().transpose(), range(whole_y.shape[1])[begin: end]


class LaplaceSmoothedMNB(BaseMNB):
    def __init__(self):
        BaseMNB.__init__(self)
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
    def _part_scoring(self, all_part_predict, real_labels, x, k=1):
        log_likelihood_mat = self.part_w.dot(x.transpose()).transpose()
        for i, each_x in enumerate(log_likelihood_mat):
            tmp = np.array(each_x.todense())[0] + self.b[real_labels[0]: real_labels[-1] + 1]
            heapq.heappush(all_part_predict[i], OnePrediction(real_labels[np.argmax(tmp)], max(tmp)))


class NonSmoothedMNB(BaseMNB):
    def _part_estimate_w(self, part_y, x):
        y_x_param = part_y.transpose().dot(x).tocsr()
        tmp = np.array(y_x_param.sum(axis=1).ravel())[0]
        y_x_param.data /= tmp.repeat(np.diff(y_x_param.indptr))
        y_x_param.data = np.log(y_x_param.data)
        return lil_matrix(y_x_param)

    def _part_scoring(self, x, k=1):
        x = x.tolil()
        labels = list()
        for sample_no in xrange(len(x.data)):
            tmp_top_labels = _one_sample_top_labels(sample_no, self.b, self.part_w, x, k)
            labels.append(tmp_top_labels)
        return labels


def _one_sample_top_labels(sample_no, b, w, x, k):
    class_scores = dict()
    x_row_tmp = x.rows[sample_no]
    x_data_tmp = x.data[sample_no]
    sample_indices_data = dict(zip(x_row_tmp, x_data_tmp))
    for label_no in xrange(w.shape[0]):
        label_no, sample_class_score = _one_label_score(label_no, b, w, sample_indices_data)
        class_scores[label_no] = sample_class_score
    return top_k_keys(class_scores, k)


def _one_label_score(label_no, b, w, one_x):
    w_data_tmp = w.data[label_no]
    w_row_tmp = w.rows[label_no]
    if len(w_data_tmp) > 0:
        label_indices_data = dict(zip(w_row_tmp, w_data_tmp))
        sample_class_score = b[label_no]
        if sample_class_score != -float("inf") and len(set(one_x.keys()).difference(set(w_row_tmp))) == 0:
            for feature in one_x.keys():
                sample_class_score += one_x[feature] * label_indices_data[feature]
            return label_no, sample_class_score
    return label_no, 1e-30


def top_k_keys(d, k):
    f = lambda x: d[x]
    sorted_d = sorted(d, key=f, reverse=True)
    return sorted_d[:k]


def _top_k_argmax(arr, k):
    if not isinstance(arr, np.ndarray):
        raise TypeError()
    for i in xrange(k):
        tmp_am = arr.argmax()
        yield tmp_am
        arr = masked_values(arr, value=arr[tmp_am])


if __name__ == '__main__':
    test_d = {'a': 2, 'b': 3, 'c': 1}
    print top_k_keys(test_d, 1)

    from preprocessing import tf_idf, transforming

    y = transforming.convert_y_to_csr(np.array([[0], [0], [1], [1], [0],
                                                [0], [0], [1], [1], [1],
                                                [1], [1], [1], [1], [0]]))
    element = [1.] * 30
    row_index = list()
    for i in range(15):
        row_index.extend([i] * 2)
    col_index = [0, 3,
                 0, 4,
                 0, 4,
                 0, 3,
                 0, 3,
                 1, 3,
                 1, 4,
                 1, 4,
                 1, 5,
                 1, 5,
                 2, 5,
                 2, 4,
                 2, 4,
                 2, 5,
                 2, 5]
    x = csr_matrix((element, (row_index, col_index)), shape=(15, 6))
    m = LaplaceSmoothedMNB()
    m.fit_and_predict(y, x)
    print x
    print "=========="
    print m.part_w
    print m._part_scoring(x)
