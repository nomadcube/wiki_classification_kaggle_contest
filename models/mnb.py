import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix


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
            self.w = csc_matrix(y_x_param)
        else:
            y_x_param = y.transpose().dot(x).tocsr()
            tmp = np.array(y_x_param.sum(axis=1).ravel())[0]
            y_x_param.data /= tmp.repeat(np.diff(y_x_param.indptr))
            y_x_param.data = np.log(y_x_param.data)
            self.w = y_x_param.tolil()
        return self

    def predict(self, x):
        x = x.tolil()
        labels = list()
        for sample_no in xrange(len(x.data)):
            sample_indices_data = {x.rows[sample_no][i]: x.data[sample_no][i] for i in xrange(len(x.data[sample_no]))}
            max_class = 0
            max_score = -1e30
            for label_no in xrange(len(self.w.data)):
                label_indices_data = {self.w.rows[label_no][i]: self.w.data[label_no][i] for i in
                                      xrange(len(self.w.data[label_no]))}
                if len(label_indices_data) == 0:
                    continue
                sample_class_score = self.b[label_no]
                if sample_class_score == -float("inf") or len(
                        set(sample_indices_data.keys()).difference(set(label_indices_data.keys()))) > 0:
                    continue
                for feature in set(sample_indices_data.keys()).intersection(set(label_indices_data.keys())):
                    sample_class_score += sample_indices_data[feature] * label_indices_data[feature]
                if sample_class_score > max_score:
                    max_score = sample_class_score
                    max_class = label_no
            labels.append([max_class])
        return labels


if __name__ == '__main__':
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
