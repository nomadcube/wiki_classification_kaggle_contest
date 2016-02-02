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

    def test_log_likelihood(self, test_y, test_x):
        model = fit_multi_label_mnb.fit(test_y, test_x)
        b = np.array([math.log(i) if i > 0 else 1e-8 for i in np.array(model[0])[0]])
        w = model[1]
        w.data = np.array([math.log(i) if i > 0 else 1e-8 for i in w.data])
        new_x = csr_matrix(([1., 1.], ([0, 0], [1, 3])), shape=(1, 6))
        ll = predict_multi_label_mnb.log_likelihood(new_x, b, w)
        assert ll.shape == (2,)
        assert abs(ll[0] - math.log(1. / 15.)) < 1e-10
        assert abs(ll[1] - math.log(1. / 45.)) < 1e-10

    def test_predict(self, test_y, test_x):
        model = fit_multi_label_mnb.fit(test_y, test_x)
        b, w = predict_multi_label_mnb.convert_to_linear(model)
        new_x = csr_matrix(([1., 1.], ([0, 0], [1, 3])), shape=(1, 6))
        predict_res = predict_multi_label_mnb.predict(new_x, b, w, 2)
        assert len(predict_res) == 1
        assert predict_res[0] == [0, 1]

    def test_top_k_argmax(self):
        a = np.array([1, 4, 5, 3, 9])
        res = [l for l in predict_multi_label_mnb.top_k_argmax(a, 2)]
        assert len(res) == 2
        assert res[0] == 4
        assert res[1] == 2
