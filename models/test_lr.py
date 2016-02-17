from array import array
import lr
import numpy as np


class TestLR:
    def pytest_funcarg__y(self):
        return [[0], [1], [0], [0], [0], [1], [0], [0], [1], [1]]

    def pytest_funcarg__x(self):
        a = [array('f', [2, 3, 3, 3, 2]), array('f', [2, 1, 1, 1, 3]), array('f', [1, 1, 2, 2, 3]),
             array('f', [3, 1, 1, 1, 2]),
             array('f', [2, 2, 3, 3, 2]), array('f', [1, 3, 2, 1, 1]), array('f', [1, 1, 2, 3, 1]),
             array('f', [1, 1, 1, 1, 3]),
             array('f', [3, 2, 1, 1, 3]), array('f', [3, 1, 2, 3, 3])]
        return np.matrix(np.array(a).reshape((10, 5)))

    def test_fit(self, y, x):
        m = lr.LR(0, 2)
        m.fit(y, x)
        assert abs(m.w[0][0] + 0.22692123) < 1e-5
        assert abs(m.w[0][1] + 15.21280659) < 1e-5
        assert abs(m.w[0][2] + (-30.07682757)) < 1e-5
        assert abs(m.w[0][3] + 14.4521558) < 1e-5
        assert abs(m.w[0][4] + 0.08459288) < 1e-5

    def test_predict(self, y, x):
        m = lr.LR(0, 2)
        m.fit(y, x)
        p = m.predict(x)
        assert p == [[0], [1], [0], [1], [0], [1], [0], [1], [1], [0]]
