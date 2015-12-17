import TrainData
import tf_idf
import math


class TestTrainData:
    def pytest_funcarg__sample_path(self):
        return '/Users/wumengling/kaggle/unit_test_data/sample.txt'

    def test_init(self, sample_path):
        tr = TrainData.TrainData(sample_path)
        assert len(tr.dat) == 3
        assert tr.instance_count == 3
        assert tr.y == {0: set([314523, 165538, 416827]),
                        1: set([21631]),
                        2: set([76255, 335416])}
        assert len(tr.x) == 3
        assert tr.x == {0: {1250536: 1},
                        1: {634175: 1, 1095476: 4, 805104: 1},
                        2: {1250536: 1, 805104: 1}}
        assert len(tr.label) == 6
        assert len(tr.label_mapping_relation) == 6
        assert len(tr.feature_set) == 4


class TestTfidf:
    def pytest_funcarg__x(self):
        return TrainData.TrainData('/Users/wumengling/kaggle/unit_test_data/sample.txt').x

    def pytest_funcarg__feature_set(self):
        return TrainData.TrainData('/Users/wumengling/kaggle/unit_test_data/sample.txt').feature_set

    def test_term_frequency(self, x):
        res = tf_idf.term_frequency(x)
        assert len(res) == len(x)
        assert res[0][1250536] == 1.0 \
               and res[1][634175] == 1.0 / 6.0 and res[1][1095476] == 4.0 / 6.0 and res[1][805104] == 1.0 / 6.0 \
               and res[2][1250536] == 0.5 and res[2][805104] == 0.5

    def test_log_inverse_doc_frequency(self, feature_set, x):
        res = tf_idf.log_inverse_doc_frequency(feature_set, x)
        assert len(res) == 4
        assert res[1250536] == math.log(3.0 / 2.0) and res[634175] == math.log(3.0 / 1.0) and \
               res[1095476] == math.log(3.0 / 1.0) and res[805104] == math.log(3.0 / 2.0)

    def test_dim_reduction_with_tf_idf(self, x, feature_set):
        res = tf_idf.dim_reduction_with_tf_idf(x, feature_set, 0.0)
        assert len(res) == 3
        assert res[0][1250536] == 1.0 * math.log(3.0 / 2.0) \
               and res[1][634175] == (1.0 / 6.0) * math.log(3.0 / 1.0) and res[1][1095476] == (4.0 / 6.0) * math.log(3.0 / 1.0) and res[1][805104] == (1.0 / 6.0) * math.log(3.0 / 2.0) \
               and res[2][1250536] == 0.5 * math.log(3.0 / 2.0) and res[2][805104] == 0.5 * math.log(3.0 / 2.0)
