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
        return np.array(a).reshape((10, 5))

    def test_fit(self, y, x):
        m = lr.LR(0, 2)
        m.fit(y, x)
        assert abs(m.w[0][0] + 0.22692079) < 1e-6
        assert abs(m.w[0][1] + 15.21590892) < 1e-6
        assert abs(m.w[0][2] + (-30.08303294)) < 1e-6
        assert abs(m.w[0][3] + 14.45525843) < 1e-6
        assert abs(m.w[0][4] + 0.0845936) < 1e-6

    def test_predict(self, y, x):
        m = lr.LR(0, 2)
        m.fit(y, x)
        p = m.predict(x)
        assert p == [[0], [1], [0], [1], [0], [1], [0], [1], [1], [0]]
