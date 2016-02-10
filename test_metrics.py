import metrics


class TestEvaluation:
    def pytest_funcarg__y(self):
        return [[314523, 165538, 416827], [21631], [76255, 335416]]

    def pytest_funcarg__predicted_y(self):
        return [[165538], [416827], [76255, 335416]]

    def pytest_funcarg__n_class(self):
        return 416827 + 1

    def test_confusion_matrix(self, y, predicted_y, n_class):
        confusion_mat = metrics.confusion_matrix(y, predicted_y, n_class)
        assert len(confusion_mat) == 4
        assert len(confusion_mat[0]) == 416828
        assert len(confusion_mat[1]) == 416828
        assert len(confusion_mat[2]) == 416828
        assert len(confusion_mat[3]) == 416828
        assert confusion_mat[0][76255] == 1
        assert confusion_mat[0][21631] == 0
        assert confusion_mat[0][335416] == 1
        assert confusion_mat[0][314523] == 0
        assert confusion_mat[0][165538] == 1
        assert confusion_mat[0][416827] == 0
        assert confusion_mat[1][76255] == 0
        assert confusion_mat[1][21631] == 1
        assert confusion_mat[1][335416] == 0
        assert confusion_mat[1][314523] == 1
        assert confusion_mat[1][165538] == 0
        assert confusion_mat[1][416827] == 1
        assert confusion_mat[2][76255] == 0
        assert confusion_mat[2][21631] == 0
        assert confusion_mat[2][335416] == 0
        assert confusion_mat[2][314523] == 0
        assert confusion_mat[2][165538] == 0
        assert confusion_mat[2][416827] == 1
        assert confusion_mat[3][76255] == 416827
        assert confusion_mat[3][21631] == 416827
        assert confusion_mat[3][335416] == 416827
        assert confusion_mat[3][314523] == 416827
        assert confusion_mat[3][165538] == 416827
        assert confusion_mat[3][416827] == 416826

    def test_macro_precision_recall(self, y, predicted_y, n_class):
        mpr = metrics.macro_precision_recall(y, predicted_y, n_class)
        assert mpr[0] == .5
        assert mpr[1] == .75
