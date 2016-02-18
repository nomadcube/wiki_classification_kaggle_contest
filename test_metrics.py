import metrics
from array import array


class TestMetrics:
    def pytest_funcarg__y(self):
        return [array('I', [65L, 66L, 67L, 68L, 69L]), array('I', [15L, 66L, 17L, 18L])]

    def pytest_funcarg__predicted_y(self):
        return [[0], [15]]

    def test_macro_precision_recall(self, y, predicted_y):
        mpr = metrics.macro_precision_recall(y, predicted_y)
        assert mpr[0] == 0.125
        assert mpr[1] == 0.125
