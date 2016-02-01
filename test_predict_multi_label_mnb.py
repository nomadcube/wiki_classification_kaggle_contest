import math

import numpy as np
from scipy.sparse import csr_matrix

import fit_multi_label_mnb
import predict_multi_label_mnb


class TestPredict:
    def pytest_funcarg__test_y(self):
        return np.array([[0], [0], [1], [1], [0],
                         [0], [0], [1], [1], [1],
                         [1], [1], [1], [1], [0]])

    def pytest_funcarg__element(self):
        return [1.] * 30

    def pytest_funcarg__row_index(self):
        row_index = list()
        for i in range(15):
            row_index.extend([i] * 2)
        return row_index

    def pytest_funcarg__col_index(self):
        return [0, 3,
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

    def pytest_funcarg__test_x(self, element, row_index, col_index):
        return csr_matrix((element, (row_index, col_index)), shape=(15, 6))

    def test_convert_to_linear_classifier(self, test_y, test_x):
        m = fit_multi_label_mnb.fit(test_y, test_x)
        m_linear = predict_multi_label_mnb.convert_to_linear_classifier(m)
        assert m_linear.shape == (2, 7)
        assert m_linear.nnz == 14
        assert abs(round(m_linear[0, 1], 12) - math.log(0.333333333333)) < 1e10
        assert abs(round(m_linear[0, 2], 12) - math.log(0.166666666667)) < 1e10
        assert abs(round(m_linear[0, 3], 12) - math.log(0.5)) < 1e10
        assert abs(round(m_linear[0, 4], 12) - math.log(0.333333333333)) < 1e10
        assert abs(round(m_linear[0, 5], 12) - math.log(0.166666666667)) < 1e10
        assert abs(round(m_linear[0, 6], 12) - math.log(0.4)) < 1e10
        assert abs(round(m_linear[1, 0], 12) - math.log(0.222222222222)) < 1e10
        assert abs(round(m_linear[1, 1], 12) - math.log(0.333333333333)) < 1e10
        assert abs(round(m_linear[1, 2], 12) - math.log(0.444444444444)) < 1e10
        assert abs(round(m_linear[1, 3], 12) - math.log(0.111111111111)) < 1e10
        assert abs(round(m_linear[1, 4], 12) - math.log(0.444444444444)) < 1e10
        assert abs(round(m_linear[1, 5], 12) - math.log(0.444444444444)) < 1e10
        assert abs(round(m_linear[1, 6], 12) - math.log(0.6)) < 1e10

    def test_log_likelihood(self, test_y, test_x):
        model = fit_multi_label_mnb.fit(test_y, test_x)
        lm = predict_multi_label_mnb.convert_to_linear_classifier(model)
        new_x = predict_multi_label_mnb.add_unit_column(csr_matrix(([1., 1.], ([0, 0], [1, 3])), shape=(1, 6)))
        ll = predict_multi_label_mnb._log_likelihood(new_x, lm)
        assert ll.shape == (2, 1)
        assert ll.nnz == 2
        assert abs(ll[0, 0] - math.log(1. / 15.)) < 1e10
        assert abs(ll[1, 0] == math.log(1. / 45.)) < 1e10

    def test_predict(self, test_y, test_x):
        model = fit_multi_label_mnb.fit(test_y, test_x)
        lm = predict_multi_label_mnb.convert_to_linear_classifier(model)
        new_x = csr_matrix(([1., 1.], ([0, 0], [1, 3])), shape=(1, 6))
        predict_res = predict_multi_label_mnb.predict(new_x, lm, 2)
        assert len(predict_res) == 1
        assert predict_res[0] == [0, 1]

    def test_top_k_argmax(self):
        a = np.array([1, 4, 5, 3, 9])
        res = [l for l in predict_multi_label_mnb._top_k_argmax(a, 2)]
        assert len(res) == 2
        assert res[0] == 4
        assert res[1] == 2
