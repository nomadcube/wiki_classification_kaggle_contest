import evaluation


class TestEvaluation:
    """Unit test for evaluation module."""

    def pytest_funcarg__y(self):
        return {0: u'314523',
                1: u'165538',
                2: u'416827',
                3: u'21631',
                4: u'76255',
                5: u'335416'}

    def pytest_funcarg__rel(self):
        return {0: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 2,
                5: 2}

    def test_generate_fact(self, y, rel):
        real_class = evaluation.generate_real_class(y, rel)
        assert isinstance(real_class, dict)
        assert len(real_class) == 6
        assert real_class[u'314523'] == {0} and real_class[u'165538'] == {0} and real_class[u'416827'] == {0} \
               and real_class[u'21631'] == {1} \
               and real_class[u'76255'] == {2} and real_class[u'335416'] == {2}

    def pytest_funcarg__fact(self, y, rel):
        return evaluation.generate_real_class(y, rel)

    def test_collect_whole_index(self, fact):
        whole_index = evaluation.all_index_in_real_class(fact)
        assert isinstance(whole_index, set)

    def pytest_funcarg__whole_index(self, fact):
        return {0, 1, 2, 3, 4, 5}

    def test_confusion_matrix(self, whole_index):
        true_index = {0, 1, 2}
        predict_index = {0, 1, 3}
        mat = evaluation.confusion_matrix(true_index, predict_index, whole_index)
        assert mat.true_pos == 2
        assert mat.false_pos == 1
        assert mat.true_neg == 2
        assert mat.false_neg == 1

    def pytest_funcarg__conf_mat(self, whole_index):
        true_index = {0, 1, 2}
        predict_index = {0, 1, 3}
        return evaluation.confusion_matrix(true_index, predict_index, whole_index)

    def test_precision_and_recall(self, conf_mat):
        res = evaluation.precision_and_recall(conf_mat)
        assert res == (2.0 / 3.0, 2.0 / 3.0)
