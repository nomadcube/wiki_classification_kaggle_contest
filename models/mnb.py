import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from functools import partial


class MNB:
    def __init__(self, alpha):
        self._alpha = alpha
        self.w = None
        self.b = None

    def fit(self, y, x):
        if not isinstance(x, csr_matrix):
            raise TypeError()
        if not isinstance(y, csr_matrix):
            raise TypeError()
        y_col_sum = np.array(y.sum(axis=0))[0]
        total_label_occurrence_cnt = y_col_sum.sum()
        y_col_sum /= total_label_occurrence_cnt
        y_col_sum = np.log(y_col_sum)
        self.b = y_col_sum

        if self._alpha != 0.:
            y_x_param = y.transpose().dot(x).todense()
            y_x_param += self._alpha
            tmp = np.array(y_x_param.sum(axis=1).ravel())[0]
            y_x_param /= tmp.repeat(y_x_param.shape[1]).reshape(y_x_param.shape)
            y_x_param = np.log(y_x_param)
            self.w = lil_matrix(y_x_param)
        else:
            y_x_param = y.transpose().dot(x).tocsr()
            tmp = np.array(y_x_param.sum(axis=1).ravel())[0]
            y_x_param.data /= tmp.repeat(np.diff(y_x_param.indptr))
            y_x_param.data = np.log(y_x_param.data)
            self.w = y_x_param.tolil()
        return self

    def predict(self, x, k=1):
        x = x.tolil()
        labels = list()
        for sample_no in xrange(len(x.data)):
            class_scores = self._one_sample_predict(x, sample_no)
            labels.append(top_k_keys(class_scores, k))
        return labels

    def _one_sample_predict(self, x, sample_no):
        class_scores = dict()
        x_row_tmp = x.rows[sample_no]
        x_data_tmp = x.data[sample_no]
        sample_indices_data = {x_row_tmp[i]: x_data_tmp[i] for i in xrange(len(x_data_tmp))}
        for label_no in xrange(1, len(self.w.data)):
            w_data_tmp = self.w.data[label_no]
            w_row_tmp = self.w.rows[label_no]
            label_indices_data = dict()
            for i in xrange(len(w_data_tmp)):
                label_indices_data[w_row_tmp[i]] = w_data_tmp[i]
            if len(label_indices_data) == 0:
                continue
            sample_class_score = self.b[label_no]
            if sample_class_score == -float("inf") or len(
                    set(sample_indices_data.keys()).difference(set(label_indices_data.keys()))) > 0:
                continue
            for feature in set(sample_indices_data.keys()).intersection(set(label_indices_data.keys())):
                sample_class_score += sample_indices_data[feature] * label_indices_data[feature]
            class_scores[label_no] = sample_class_score
        return class_scores


def top_k_keys(d, k):
    if not isinstance(d, dict):
        raise TypeError()
    sorted_d = sorted(d, key=lambda x: d[x], reverse=True)
    return sorted_d[:k]


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
    m = MNB(0.)
    m.fit(y, x)
    print m.w
    print m.predict(x)
