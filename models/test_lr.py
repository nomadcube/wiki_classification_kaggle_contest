from array import array
import lr


class TestLR:
    def pytest_funcarg__y(self):
        return [0, 1, 0, 0, 0, 1, 0, 0, 1, 1]

    def pytest_funcarg__x(self):
        return [array('f', [2, 3, 3, 3, 2]), array('f', [2, 1, 1, 1, 3]), array('f', [1, 1, 2, 2, 3]),
                array('f', [3, 1, 1, 1, 2]),
                array('f', [2, 2, 3, 3, 2]), array('f', [1, 3, 2, 1, 1]), array('f', [1, 1, 2, 3, 1]),
                array('f', [1, 1, 1, 1, 3]),
                array('f', [3, 2, 1, 1, 3]), array('f', [3, 1, 2, 3, 3])]

    def test_fit(self, y, x):
        m = lr.LR(0, 2)
        m.fit(y, x)
        assert abs(m.w[0] - 0.22692853) < 1e-6
        assert abs(m.w[1] - 16.43603803) < 1e-6
        assert abs(m.w[2] - (-32.52328342)) < 1e-6
        assert abs(m.w[3] - 15.67538028) < 1e-6
        assert abs(m.w[4] - 0.084586) < 1e-6

    def test_predict(self, y, x):
        m = lr.LR(0, 2)
        m.fit(y, x)
        p = m.predict(x)
        assert p == [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
