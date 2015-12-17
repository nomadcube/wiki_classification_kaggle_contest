import evaluation


class TestEvaluation:
    """Unit test for evaluation module."""

    def pytest_funcarg__test_data_path(self):
        return '/Users/wumengling/kaggle/unit_test_data/sample.txt'

    def pytest_funcarg__predict_data_path(self):
        return '/Users/wumengling/kaggle/unit_test_data/predict.txt'

    def test_real_label_id_map(self, test_data_path):
        label_id = evaluation.true_id_per_label(test_data_path)
        assert isinstance(label_id, dict)
        assert len(label_id) == 6
        assert label_id['314523'] == {0} and label_id['165538'] == {0} and label_id['416827'] == {0} \
               and label_id['21631'] == {1} \
               and label_id['76255'] == {2} and label_id['335416'] == {2}

    def pytest_funcarg__real_label_id_map(self, test_data_path):
        return evaluation.true_id_per_label(test_data_path)

    def test_each_label_metric(self, predict_data_path, real_label_id_map):
        label_measure = evaluation.confusion_matrix_per_label(predict_data_path, real_label_id_map)
        assert isinstance(label_measure, dict)
        assert len(label_measure) == 6
        assert label_measure['314523'].true_pos == 1.0
        assert label_measure['314523'].false_pos == 0.0
        assert label_measure['314523'].true_neg == 2.0
        assert label_measure['314523'].false_neg == 0.0

    def test_macro_metric(self, test_data_path, predict_data_path):
        res = evaluation.macro_metric(test_data_path, predict_data_path)
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res == (2.5 / 6.0, 0.5)
