import evaluation_metrics


class TestEvaluation:
    """Unit test for evaluation module."""

    def pytest_funcarg__y(self):
        return ['314523,165538,416827', '21631', '76255,335416']

    def test_generate_fact(self, y):
        real_class = evaluation_metrics.index_for_each_label(y)
        assert isinstance(real_class, dict)
        assert len(real_class) == 6
        assert real_class[u'314523'] == {0} and real_class[u'165538'] == {0} and real_class[u'416827'] == {0} \
               and real_class[u'21631'] == {1} \
               and real_class[u'76255'] == {2} and real_class[u'335416'] == {2}

    def pytest_funcarg__fact(self, y):
        return evaluation_metrics.index_for_each_label(y)

    def test_collect_whole_index(self, fact):
        whole_index = evaluation_metrics.all_index(fact)
        assert isinstance(whole_index, set)

    def pytest_funcarg__whole_index(self, fact):
        return {0, 1, 2, 3, 4, 5}

    def test_confusion_matrix(self, whole_index):
        true_index = {0, 1, 2}
        predict_index = {0, 1, 3}
        mat = evaluation_metrics.confusion_matrix(true_index, predict_index, whole_index)
        assert mat.true_pos == 2
        assert mat.false_pos == 1
        assert mat.true_neg == 2
        assert mat.false_neg == 1

    def pytest_funcarg__conf_mat(self, whole_index):
        true_index = {0, 1, 2}
        predict_index = {0, 1, 3}
        return evaluation_metrics.confusion_matrix(true_index, predict_index, whole_index)

    def test_precision_and_recall(self, conf_mat):
        res = evaluation_metrics.precision_and_recall(conf_mat)
        assert res == (2.0 / 3.0, 2.0 / 3.0)

    def test_macro_precision_and_recall(self, y):
        prediction = ['314523,165538,416827', '21631', '76255,335416']
        assert evaluation_metrics.macro_precision_and_recall(y, prediction) == (1, 1)
        prediction = ['314523,165538,416827', '314523,165538,416827', '76255,335416']
        assert evaluation_metrics.macro_precision_and_recall(y, prediction) == (3.5 / 6, 5.0 / 6)
