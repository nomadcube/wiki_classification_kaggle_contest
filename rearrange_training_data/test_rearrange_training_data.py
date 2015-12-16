import math
import tf_idf
import libsvm_train_data
import label_mapping


class TestRearrangeTrainingData:
    def pytest_funcarg__sample_path(self):
        return '/Users/wumengling/kaggle/unit_test_data/sample.txt'

    def test_log_inverse_doc_frequency(self, sample_path):
        res = tf_idf.log_inverse_doc_frequency(sample_path)
        assert isinstance(res, dict)
        assert len(res) == 4
        assert res['1250536'] == math.log(3.0 / 2.0) and res['634175'] == math.log(3.0 / 1.0) and \
               res['1095476'] == math.log(3.0 / 1.0) and res['805104'] == math.log(3.0 / 2.0)

    def test_term_frequency(self, sample_path):
        res = tf_idf.term_frequency(sample_path)
        assert isinstance(res, dict)
        assert res[0]['1250536'] == 1.0 \
               and res[1]['634175'] == 1.0 / 6.0 and res[1]['1095476'] == 4.0 / 6.0 and res[1]['805104'] == 1.0 / 6.0 \
               and res[2]['1250536'] == 0.5 and res[2]['805104'] == 0.5

    def test_training_label(self, sample_path):
        res = [i for i in libsvm_train_data.training_label(sample_path)]
        assert isinstance(res, list)
        assert len(res) == 3
        assert res == [314523165538416827, 21631, 76255335416]

    def test_dimension_reduction_instance(self, sample_path):
        res = [i for i in libsvm_train_data.dimension_reduction_instance(tf_idf.term_frequency(sample_path), tf_idf.log_inverse_doc_frequency(sample_path), 0)]
        assert isinstance(res, list)
        assert len(res) == 3
        assert res[0][1250536] == 1.0 * math.log(3.0 / 2.0) \
               and res[1][634175] == (1.0 / 6.0) * math.log(3.0 / 1.0) and res[1][1095476] == (4.0 / 6.0) * math.log(3.0 / 1.0) and res[1][805104] == (1.0 / 6.0) * math.log(3.0 / 2.0) \
               and res[2][1250536] == 0.5 * math.log(3.0 / 2.0) and res[2][805104] == 0.5 * math.log(3.0 / 2.0)

    def test_label(self, sample_path):
        res = label_mapping.label(sample_path)
        assert isinstance(res, set)
        assert ('314523,165538,416827' in res) and ('21631' in res) and ('76255,335416' in res)

    def pytest_funcarg__label_set(self, sample_path):
        return label_mapping.label(sample_path)

    def test_label_mapping(self, label_set):
        res = label_mapping.label_mapping(label_set)
        assert len(res) == 3
        val = [v for v in res.keys()]
        assert ('314523,165538,416827' in val) and ('21631' in val) and ('76255,335416' in val)
        assert [k for k in res.values()] == [0, 1, 2]
