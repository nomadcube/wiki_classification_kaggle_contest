import evaluation


class TestEvaluation:
    def pytest_funcarg__y(self):
        return [[314523, 165538, 416827], [21631], [76255, 335416]]

    def pytest_funcarg__predicted_y(self):
        return [[165538], [416827], [76255, 335416]]

    def pytest_funcarg__n_class(self):
        return 416827 + 1

    def test_confusion_matrix(self, y, predicted_y, n_class):
        cm = evaluation.confusion_matrix(y, predicted_y, n_class)
        assert len(cm) == 4
        assert len(cm[0]) == 416828
        assert len(cm[1]) == 416828
        assert len(cm[2]) == 416828
        assert len(cm[3]) == 416828
        assert cm[0][76255, 0] == 1
        assert cm[0][21631, 0] == 0
        assert cm[0][335416, 0] == 1
        assert cm[0][314523, 0] == 0
        assert cm[0][165538, 0] == 1
        assert cm[0][416827, 0] == 0
        assert cm[1][76255, 0] == 0
        assert cm[1][21631, 0] == 1
        assert cm[1][335416, 0] == 0
        assert cm[1][314523, 0] == 1
        assert cm[1][165538, 0] == 0
        assert cm[1][416827, 0] == 1
        assert cm[2][76255, 0] == 0
        assert cm[2][21631, 0] == 0
        assert cm[2][335416, 0] == 0
        assert cm[2][314523, 0] == 0
        assert cm[2][165538, 0] == 0
        assert cm[2][416827, 0] == 1
        assert cm[3][76255, 0] == 2
        assert cm[3][21631, 0] == 2
        assert cm[3][335416, 0] == 2
        assert cm[3][314523, 0] == 2
        assert cm[3][165538, 0] == 2
        assert cm[3][416827, 0] == 1

    def test_precision_recall(self, y, predicted_y, n_class):
        pr = evaluation.precision_recall(y, predicted_y, n_class)
        assert len(pr) == 2
        assert pr[0][76255] == 1.
        assert pr[0][21631] == 0.
        assert pr[0][335416] == 1.
        assert pr[0][314523] == 0.
        assert pr[0][165538] == 1.
        assert pr[0][416827] == 0.
        assert pr[1].mask[0].tolist().count(False) == 4
        assert pr[1][0, 76255] == 1.
        assert pr[1][0, 335416] == 1.
        assert pr[1][0, 165538] == 1.
        assert pr[1][0, 416827] == 0.

    def test_macro_precision_recall(self, y, predicted_y, n_class):
        mpr = evaluation.macro_precision_recall(y, predicted_y, n_class)
        assert mpr[0] == .5
        assert mpr[1] == .75
